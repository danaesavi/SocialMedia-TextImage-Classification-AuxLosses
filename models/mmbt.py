import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from tqdm import tqdm
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    MMBTConfig,
    AutoTokenizer
)

from transformers.models.mmbt.modeling_mmbt import MMBTModel
from utils import (
    prepare_data,
    agg_metrics_val,
    get_optimizer_params,
    clip_loss,
    get_conv_model,
    loss_correction
)

from config import (
    img_feat_size_cnn,
    fixed_feat_size,
    MODEL_DIR_DICT,
    metric_names,
    T
)
from datasets import MMBT_Dataset
from torch.utils.data import DataLoader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ------ LOGGING-----------------------------------------------------
import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S',
                   level=logging.INFO,
                   )
logger = logging.getLogger(__name__)

# ------- MODELS ------------------------------------------------------

POOLING_BREAKDOWN = {1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (2, 2), 5: (5, 1), 6: (3, 2), 7: (7, 1), 8: (4, 2), 9: (3, 3)}

class ImageEncoder(nn.Module):
    def __init__(self, num_image_embeds):
        super().__init__()
        self.img_model_name = "resnet152"
        img_model_dir = MODEL_DIR_DICT[self.img_model_name]
        original_img_model = get_conv_model(self.img_model_name)
        original_img_model.load_state_dict(torch.load(img_model_dir))
        modules = list(original_img_model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d(POOLING_BREAKDOWN[num_image_embeds])

    def forward(self, x):
        # Bx3x224x224 -> Bx2048x7x7 -> Bx2048xN -> BxNx2048
        out = self.pool(self.model(x))
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out  # BxNx2048

################## MMBT ################################################

class MMBT(nn.Module):

    def __init__(self, num_labels, txt_model_name, dropout, num_image_embeds=1, 
    logit_scale_init_value=2.6592, multilabel=False):
        super().__init__()
        self.num_labels = num_labels
        self.multilabel = multilabel
        self.txt_model_name = txt_model_name
        txt_model_dir = MODEL_DIR_DICT[self.txt_model_name]
        self.transformer =  AutoModel.from_pretrained(txt_model_dir).to(device)
        transformer_config = AutoConfig.from_pretrained(txt_model_dir)
        config = MMBTConfig(transformer_config, num_labels=self.num_labels)
        self.encoder = ImageEncoder(num_image_embeds).to(device)
        self.mmbt = MMBTModel(config, self.transformer, self.encoder).to(device)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fixed_feat_size, self.num_labels)
        self.proj_embeddings = nn.Linear(img_feat_size_cnn, fixed_feat_size)
        self.linear_tim = nn.Linear(fixed_feat_size, 2)
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)

    def forward(self, inputs, tim_inputs=None, itc=False):

        outputs = self.mmbt(**inputs)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if itc:
            # for ITC
            x_v = self.proj_embeddings(self.encoder(inputs["input_modal"])) # BXNX768
            # batch_size, _ , _ = x_v.size()
            # x_v = x_v.view(batch_size,fixed_feat_size) # BX768*num_image_embeds
            x_v = torch.squeeze(x_v, 1)
            out_xt = self.transformer(
                input_ids = inputs["input_ids"],
                attention_mask = inputs["attention_mask"],
                ) # text inputs
            x_t = out_xt.last_hidden_state[:,0,:]
            
        else:
            x_v, x_t = None, None

        if tim_inputs!= None:
            #for ITM: run mmbt with other inputs
            out_tim = self.mmbt(**tim_inputs)
            out_tim = outputs[1]
            out_tim = self.linear_tim(out_tim)
        else:
            out_tim = None

        return logits, x_t, x_v, out_tim
    
    def get_logits_per_text(self, text_embeds, image_embeds):
        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
       
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        return logits_per_text


class MMBT_Model(object):
    """
    MMBT_Model class
    """       
    
    def __init__(self, config, txt_model_name='bert', multilabel=False):
        """ Initialization """
        self.batch_size = config.batch_size
        self.num_labels = config.num_labels
        self.multilabel = multilabel
        self.max_length = config.max_length
        # Aux tasks
        self.use_clip_loss = config.use_clip_loss
        self.beta_itc = config.beta_itc
        self.use_tim_loss = config.use_tim_loss
        self.beta_itm = config.beta_itm
        # LC
        self.use_loss_correction = config.use_loss_correction
        # Model
        self.model = MMBT(self.num_labels, txt_model_name, config.dropout)
        self.model.to(device)
        #print(self.model)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        # tokenizer
        txt_model_dir =  txt_model_dir = MODEL_DIR_DICT[txt_model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(txt_model_dir)
    
    def collate_fn(self, batch):
        lens = [len(row["sentence"]) for row in batch]
        bsz, max_seq_len = len(batch), max(lens)

        mask_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)
        text_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)

        for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
            text_tensor[i_batch, :length] = input_row["sentence"]
            mask_tensor[i_batch, :length] = 1

        img_tensor = torch.stack([row["image"] for row in batch])
        tgt_tensor = torch.stack([row["label"] for row in batch])
        img_start_token = torch.stack([row["image_start_token"] for row in batch])
        img_end_token = torch.stack([row["image_end_token"] for row in batch])
        data_ids = torch.stack([row["data_id"] for row in batch])

        return text_tensor, mask_tensor, img_tensor, img_start_token, img_end_token, tgt_tensor, data_ids
    
    def load_data(self, data, img_file_fmt, testing = False, task_name=None):
        train, y_vector_tr, val, y_vector_val, test, y_vector_te, class_weights, image_adds = prepare_data(
            data, self.num_labels, testing=testing, multilabel=self.multilabel)

        tr_dataset = MMBT_Dataset(
            train.tweet_id.values,train.text.values,y_vector_tr,
            self.tokenizer, self.max_length, img_file_fmt, task_name=task_name)
        val_dataset = MMBT_Dataset(
            val.tweet_id.values,val.text.values,y_vector_val,
            self.tokenizer, self.max_length, img_file_fmt, task_name=task_name)
        te_dataset = MMBT_Dataset(
            test.tweet_id.values,test.text.values,y_vector_te,
            self.tokenizer, self.max_length, img_file_fmt, task_name=task_name)
        
        train_loader = DataLoader(tr_dataset, collate_fn=self.collate_fn, 
        batch_size=self.batch_size, shuffle=True) 
        val_loader = DataLoader(val_dataset, collate_fn=self.collate_fn, 
        batch_size=self.batch_size,shuffle=False)        
        test_loader = DataLoader(te_dataset, collate_fn=self.collate_fn, 
        batch_size=self.batch_size,shuffle=False) 

        return train_loader, val_loader ,test_loader, class_weights
    
    def prepare_itm_inputs(self, inputs):
        ids, mask = inputs["input_ids"], inputs["attention_mask"]
        m_batchsize, _ = ids.size()
        #print("batch size", m_batchsize)
        if m_batchsize>1:

            # replace text with 0.5 probability
            # make copies of text info
            tim_ids = ids.clone().detach()
            tim_mask = mask.clone().detach()
            labels_tim = []
            for idx in range(m_batchsize):
                change_text = np.random.choice([True, False])
                if change_text:
                    # mismatch
                    labels_tim.append(0) 
                    indexes = set(range(m_batchsize)) - {idx}
                    # choose a diff example within the batch and change
                    new_idx = np.random.choice(list(indexes))
                    tim_ids[idx] = ids[new_idx]
                    tim_mask[idx] = mask[new_idx]
                else:
                    # match
                    labels_tim.append(1)
            tim_ids = tim_ids.to(device)
            tim_mask = tim_mask.to(device)
            lbl_tim = torch.tensor(labels_tim, dtype=torch.long).to(device)

            tim_inputs = {
                "input_ids": tim_ids,
                "input_modal": inputs["input_modal"],
                "attention_mask": tim_mask,
                "modal_start_tokens": inputs["modal_start_tokens"],
                "modal_end_tokens": inputs["modal_end_tokens"],
                "return_dict": False
            }
        else:
            tim_inputs = inputs
            lbl_tim = torch.tensor([1], dtype=torch.long).to(device)
        
        return tim_inputs, lbl_tim
    
    def train(self,dataloader,val_dataloader,epochs,loss_fn,lr,weight_decay,
        tim_loss_fn = None, te_dataloader=None,model_path=None,val_filename=None,
        te_filename=None):
        #Initialize Optimizer
        named_parameters = self.model.named_parameters()
        optimizer_params = get_optimizer_params(named_parameters, weight_decay, lr)
        optimizer = optim.AdamW(optimizer_params, lr=lr)        
        res_val, res_te = [], []
        for epoch in range(epochs):
            logger.info("Epoch: " + str(epoch+1))
            for batch in tqdm(dataloader):
                self.model.train()
                batch = tuple(t.to(device) for t in batch)
                label = batch[5]
                inputs = {
                    "input_ids": batch[0],
                    "input_modal": batch[2],
                    "attention_mask": batch[1],
                    "modal_start_tokens": batch[3],
                    "modal_end_tokens": batch[4],
                    "return_dict": False
                }
    
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                if self.use_tim_loss:
                    # TIM task
                    tim_inputs, lbl_tim =self.prepare_itm_inputs(inputs)
                    output, x_t, x_v, output_tim = self.model(
                        inputs,
                        tim_inputs=tim_inputs,
                        itc=self.use_clip_loss
                        )
                else:
                    output, x_t, x_v, _ = self.model(inputs, itc=self.use_clip_loss)
                
                label = label.type_as(output)            
                # compute loss
                if self.use_clip_loss and self.use_tim_loss:
                    logits_per_text = self.model.get_logits_per_text(x_t,x_v)
                    loss = (1-(self.beta_itc+self.beta_itm)) * loss_fn(output,label) + self.beta_itc * clip_loss(logits_per_text) + self.beta_itm * tim_loss_fn(output_tim,lbl_tim)                  
                elif self.use_clip_loss:
                    logits_per_text = self.model.get_logits_per_text(x_t,x_v)
                    loss = (1-self.beta_itc) * loss_fn(output,label) + self.beta_itc * clip_loss(logits_per_text)  
                elif self.use_tim_loss:
                    #print("out", type(output_tim))
                    #print("label tim", type(lbl_tim))
                    loss = (1-self.beta_itm) * loss_fn(output,label) + self.beta_itm * tim_loss_fn(
                        output_tim,lbl_tim)                  
                elif self.use_loss_correction:
                    loss = loss_correction(T,loss_fn, output, label)
                else:
                    loss = loss_fn(output,label)
                # backward loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # predict
                # pred = torch.argmax(self.softmax(output),dim=1)  
                # target = torch.argmax(label,dim = 1)          
                # num_correct = torch.sum(pred==target).item()
                # num_samples = label.size(0)
                # logger.info(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
            
            # predict val
            logger.info("val")
            res_val_d = self.eval(val_dataloader,loss_fn,tim_loss_fn=tim_loss_fn)
            res_val_d["epoch"] = epoch
            res_val.append(res_val_d)
            if val_filename != None and (epoch%2 == 0 or epoch==epochs-1):
                logger.info("Compute metrics (val)")
                metrics_val = agg_metrics_val(res_val, metric_names, self.num_labels)
                pd.DataFrame(metrics_val).to_csv(val_filename,index=False)
                logger.info("{} saved!".format(val_filename))

            if te_dataloader != None:
                # predict test
                res_te_d = self.eval(te_dataloader,loss_fn,tim_loss_fn=tim_loss_fn)
                res_te_d["epoch"] = epoch
                res_te.append(res_te_d)
                if te_filename != None and (epoch%2 == 0 or epoch==epochs-1):
                    metrics_te = agg_metrics_val(res_te, metric_names, self.num_labels)
                    pd.DataFrame(metrics_te).to_csv(te_filename,index=False)
                    logger.info("{} saved!".format(te_filename))

        if model_path != None:
            torch.save(self.model.state_dict(), model_path)
            logger.info("{} saved".format(model_path))
    
    def eval(self, dataloader, loss_fn, tim_loss_fn=None):
        eval_acc = []
        eval_loss = []
        predictions = []
        labels = []
        data_ids = []
        self.model.eval()

        for batch in tqdm(dataloader):
            batch = tuple(t.to(device) for t in batch)
            label = batch[5]
            data_id = batch[6]
            inputs = {
                "input_ids": batch[0],
                "input_modal": batch[2],
                "attention_mask": batch[1],
                "modal_start_tokens": batch[3],
                "modal_end_tokens": batch[4],
                "return_dict": False
            }
            
            # Compute logits
            with torch.no_grad():
                if self.use_tim_loss:
                    # TIM task
                    tim_inputs, lbl_tim =self.prepare_itm_inputs(inputs)
                    output, x_t, x_v, output_tim = self.model(
                        inputs,
                        tim_inputs=tim_inputs,
                         itc=self.use_clip_loss
                        )
                else:
                    output, x_t, x_v, _ = self.model(inputs, itc=self.use_clip_loss)
                
                label = label.type_as(output)               
                # compute loss
                if self.use_clip_loss and self.use_tim_loss:
                    logits_per_text = self.model.get_logits_per_text(x_t,x_v)
                    loss = (1-(self.beta_itc+self.beta_itm)) * loss_fn(output,label) + self.beta_itc * clip_loss(logits_per_text) + self.beta_itm * tim_loss_fn(output_tim,lbl_tim)                  
                elif self.use_clip_loss:
                    logits_per_text = self.model.get_logits_per_text(x_t,x_v)
                    loss = (1-self.beta_itc) * loss_fn(output,label) + self.beta_itc * clip_loss(
                        logits_per_text)   
                elif self.use_tim_loss:
                    loss = (1-self.beta_itm) * loss_fn(output,label) + self.beta_itm * tim_loss_fn(
                        output_tim,lbl_tim)                  
                elif self.use_loss_correction:
                    loss = loss_correction(T,loss_fn, output, label)
                else:
                    loss = loss_fn(output,label)

            eval_loss.append(loss.item())
            # Get the predictions
            if not self.multilabel:
                pred = torch.argmax(self.softmax(output), dim=1)  
                target = torch.argmax(label,dim = 1)  
            else:
                soft_pred = self.sigmoid(output)
                pred = torch.round(soft_pred)
                target = label
            # Calculate the accuracy rate
            accuracy = (pred == target).cpu().numpy().mean() * 100
            eval_acc.append(accuracy)
            # Save predictions and targets
            predictions += pred
            labels += target
            data_ids += data_id
        
        # Compute the average accuracy and loss over the validation set.
        eval_loss = np.mean(eval_loss)
        eval_acc = np.mean(eval_acc)

        logger.info(f'loss: {eval_loss:.4f} acc: {(eval_acc):.4f}\n')
        
        y_pred = torch.stack(predictions)
        y = torch.stack(labels)
        data_ids = torch.stack(data_ids)
        
        res = {
            "data_id": data_ids,
            "loss": eval_loss,
            "predictions": y_pred,
            "labels": y
        }
        
        return res

    

    
    



        
        
        