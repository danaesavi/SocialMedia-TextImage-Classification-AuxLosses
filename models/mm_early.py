import sys
sys.path.append("../")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from lxmert_scripts.modeling_frcnn import GeneralizedRCNN
from lxmert_scripts.utils import Config
from lxmert_scripts.processing_image import Preprocess
from transformers import LxmertTokenizer, LxmertModel
from transformers import ViltModel, ViltProcessor
from datasets import ViLT_Dataset, Lxmert_Dataset
from utils import (
    prepare_data,
    agg_metrics_val,
    get_optimizer_params,
    clip_loss,
    loss_correction
)
from config import (
    img_feat_size,
    txt_feat_size,
    fixed_feat_size,
    MODEL_DIR_DICT,
    metric_names,
    T
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ------ LOGGING-----------------------------------------------------
import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S',
                   level=logging.INFO,
                   )
logger = logging.getLogger(__name__)
# --------MODELS -------------------------------------------------------

def get_early_model(model_name):
     MODEL_DICT = {
        'vilt': ViLT,
        'lxmert': Lxmert,
     }
     return MODEL_DICT[model_name]

class ViLT(nn.Module):
    def __init__(self, model_dir, num_labels, max_length, dropout=0.1,logit_scale_init_value=2.6592):
        super(ViLT, self).__init__()
        self.max_length = max_length
        self.model = ViltModel.from_pretrained(model_dir, max_position_embeddings=self.max_length) #, ignore_mismatched_sizes=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(img_feat_size, num_labels) # 768 -> num_labels
        self.visual_projection = nn.Linear(img_feat_size, fixed_feat_size, bias=False)
        self.text_projection = nn.Linear(txt_feat_size, fixed_feat_size, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)
        self.linear_tim = nn.Linear(fixed_feat_size, 2)

    def forward(self,ids,mask,token_type_ids,pixel_values,pixel_mask,tim_inputs=None):
        
        last_hidden, pooled_output= self.model(
            input_ids=ids, 
            attention_mask=mask, 
            token_type_ids=token_type_ids,
            pixel_mask=pixel_mask, 
            pixel_values = pixel_values, 
            return_dict=False
            )
        xt_xv = self.dropout(last_hidden[:,0,:])
        linear_output = self.linear(xt_xv)

        # ITC split up final hidden states into text and image features for contrastive loss
        text_seq_len = ids.shape[1] 
        x_t, x_v = (last_hidden[:,0,:], last_hidden[:, text_seq_len,:])

        # TIM
        if tim_inputs != None:
            tim_ids, tim_mask, tim_token_type_ids = tim_inputs
            last_hidden_tim, pooled_output_tim= self.model(
                input_ids=tim_ids, 
                attention_mask=tim_mask, 
                token_type_ids=tim_token_type_ids,
                pixel_mask=pixel_mask, 
                pixel_values = pixel_values, 
                return_dict=False
                )

            xt_xv_tim = last_hidden_tim[:,0,:]
            out_tim =self.linear_tim(xt_xv_tim)
        else:
            out_tim = None

        return linear_output, x_t, x_v, out_tim
    
    def get_logits_per_text(self, text_embeds,image_embeds):
        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        return logits_per_text
 
class Lxmert(nn.Module):
    def __init__(self, model_dir, num_labels, max_length=None, dropout=0.1, logit_scale_init_value=2.6592):
        '''
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* parameter. 
            Default is used as per the original CLIP implementation.
        '''
        super(Lxmert, self).__init__()
        self.model = LxmertModel.from_pretrained(model_dir)
        self.linear_fusion = nn.Linear(fixed_feat_size, fixed_feat_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(fixed_feat_size, num_labels) # 768 -> num_labels
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)
        self.linear_tim = nn.Linear(fixed_feat_size, 2)

    def forward(self,ids,mask,token_type_ids,features,normalized_boxes,tim_inputs=None):
        output = self.model(
         input_ids=ids,
         attention_mask=mask,
         visual_feats=features,
         visual_pos=normalized_boxes,
         token_type_ids=token_type_ids,
         output_attentions=False,
         )
        x_t = output.language_output
        x_v = output.vision_output
        # the crossmodal output in lxmert is the first token of the language output
        xt_xv = x_t[:, 0, :]
        xt_xv = self.relu(self.linear_fusion(xt_xv))
        xt_xv = self.dropout(xt_xv)
        linear_output = self.linear(xt_xv)

        # max pooling to get image and text embeds for contrastive loss
        input_mask_expanded = mask.unsqueeze(-1).expand(x_t.size()).float()
        last_hidden_t = x_t.clone().detach()
        last_hidden_t[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        max_embeddings_t = torch.max(last_hidden_t, 1)[0]
        max_embeddings_v = torch.max(x_v,1)[0]

        #TIM
        if tim_inputs != None:
            tim_ids, tim_mask, tim_token_type_ids = tim_inputs
            lxm_tim = self.model(
                    input_ids=tim_ids,
                    attention_mask=tim_mask,
                    visual_feats=features,
                    visual_pos=normalized_boxes,
                    token_type_ids=tim_token_type_ids,
                    output_attentions=False,
                    )
            # the crossmodal output in lxmert is the first token of the language output
            xt_xv_tim = lxm_tim.language_output[:, 0, :]
            out_tim =self.linear_tim(xt_xv_tim)
        else:
            out_tim = None
            
        
        return linear_output, max_embeddings_t, max_embeddings_v, out_tim

    def get_logits_per_text(self, text_embeds,image_embeds):
        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        return logits_per_text


class MMEarly_Model(object):
    """
    MMEarly_Model class
    """
       
    def __init__(self, config, model_name,multilabel=False):
        """ Initialization """
        self.batch_size = config.batch_size
        self.num_labels = config.num_labels
        self.multilabel = multilabel
        self.use_clip_loss = config.use_clip_loss
        self.beta_itc = config.beta_itc
        self.use_tim_loss = config.use_tim_loss
        self.beta_itm = config.beta_itm
        self.use_loss_correction = config.use_loss_correction
        self.model_name = model_name
        self.model_dir = MODEL_DIR_DICT[self.model_name]
        self.max_length = config.max_length
        # preprocessing 
        if self.model_name=="vilt":
            self.processor = ViltProcessor.from_pretrained(self.model_dir) #,data_format="channels_first") # max_length=self.max_length
        else:
            self.tokenizer = LxmertTokenizer.from_pretrained(self.model_dir)
        # model
        self.model = get_early_model(model_name)(self.model_dir, self.num_labels, self.max_length, dropout=config.dropout)
        self.model.to(device)
        #print(self.model)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
    
    def collate_fn(self,batch):
        #print("item", batch[0].keys())
        input_ids = [torch.squeeze(item['input_ids']) for item in batch]
        pixel_values = [torch.squeeze(item['pixel_values']) for item in batch]
        attention_mask = [torch.squeeze(item['attention_mask']) for item in batch]
        token_type_ids = [torch.squeeze(item['token_type_ids']) for item in batch]
        labels = [item['labels'] for item in batch]
        data_ids = [item['data_id'] for item in batch]
        
        # create padded pixel values and corresponding pixel mask
        encoding = self.processor.feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
        
        # create new batch
        batch = {}
        batch['input_ids'] = torch.stack(input_ids)
        batch['attention_mask'] = torch.stack(attention_mask)
        batch['token_type_ids'] = torch.stack(token_type_ids)
        batch['pixel_values'] = encoding['pixel_values']
        batch['pixel_mask'] = encoding['pixel_mask']
        batch['labels'] = torch.stack(labels)
        batch['data_id'] = torch.stack(data_ids)
        return batch
    
    def load_data(self, data, img_file_fmt=None, task_name=None, testing=False,
    saved_features=False):

        train, y_vector_tr, val, y_vector_val, test, y_vector_te, class_weights, image_adds = prepare_data(
            data, self.num_labels, testing=testing, multilabel=self.multilabel)        
        if self.model_name == "vilt":
            # train
            tr_dataset = ViLT_Dataset(train.tweet_id.values,train.text.values,y_vector_tr,
            self.processor, self.max_length, img_file_fmt,
            saved_features=saved_features,
            task_name=task_name)
            train_loader = DataLoader(tr_dataset, collate_fn=self.collate_fn, batch_size=self.batch_size, shuffle=True)
            # val
            val_dataset = ViLT_Dataset(val.tweet_id.values, val.text.values,y_vector_val,
            self.processor, self.max_length, img_file_fmt,
            saved_features=saved_features,
            task_name=task_name)
            val_loader = DataLoader(val_dataset, collate_fn=self.collate_fn, batch_size=self.batch_size, shuffle=False)
            # test
            te_dataset = ViLT_Dataset(test.tweet_id.values, test.text.values,y_vector_te,
            self.processor, self.max_length, img_file_fmt,
            saved_features=saved_features,
            task_name=task_name)
            test_loader = DataLoader(te_dataset, collate_fn=self.collate_fn, batch_size=self.batch_size, shuffle=False)
        else:
            tr_dataset = Lxmert_Dataset(train.tweet_id.values,train.text.values,y_vector_tr,self.tokenizer, self.max_length, task_name)
            train_loader = DataLoader(tr_dataset, batch_size=self.batch_size, shuffle=True)
            val_dataset = Lxmert_Dataset(val.tweet_id.values, val.text.values,y_vector_val,self.tokenizer, self.max_length, task_name)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            te_dataset = Lxmert_Dataset(test.tweet_id.values, test.text.values,y_vector_te,self.tokenizer, self.max_length, task_name)
            test_loader = DataLoader(te_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader, class_weights

    def prepare_itm_inputs(self, ids, mask, token_type_ids):
        # replace text with 0.5 probability
        # make copies of text info
        tim_ids = ids.clone().detach()
        tim_mask = mask.clone().detach()
        tim_token_type_ids = token_type_ids.clone().detach()
        labels_tim = []
        m_batchsize, _ = ids.size()
        #print("batch size", m_batchsize)
        if m_batchsize>1:
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
                    tim_token_type_ids[idx] = token_type_ids[new_idx]
                else:
                    # match
                    labels_tim.append(1)
        else:
            labels_tim.append(1)

        tim_ids = tim_ids.to(device)
        tim_mask = tim_mask.to(device)
        tim_token_type_ids = tim_token_type_ids.to(device)
        lbl_tim = torch.tensor(labels_tim, dtype=torch.long).to(device)
        return tim_ids, tim_mask, tim_token_type_ids, lbl_tim

    def train(self,dataloader,val_dataloader,epochs,loss_fn,lr,weight_decay,
    tim_loss_fn = None, te_dataloader=None,model_path=None,val_filename=None,
    te_filename=None):
        #Initialize Optimizer
        named_parameters = self.model.named_parameters()
        optimizer_params = get_optimizer_params(named_parameters, weight_decay, lr)
        optimizer = optim.AdamW(optimizer_params, lr=lr)
        #optimizer= optim.Adam(self.model.parameters(), lr= lr, weight_decay = weight_decay)
        
        self.model.train()
        res_val, res_te = [], []
        for epoch in range(epochs):
            logger.info("Epoch: " + str(epoch+1))
            for batch in tqdm(dataloader):
                ids = torch.squeeze(batch['input_ids']).to(device)
                mask = torch.squeeze(batch['attention_mask']).to(device)
                token_type_ids = torch.squeeze(batch['token_type_ids']).to(device)
                if self.model_name == "vilt":
                    pixel_mask = torch.squeeze(batch['pixel_mask']).to(device)
                    pixel_values = torch.squeeze(batch['pixel_values']).to(device)
                    if len(pixel_values.size())<4:
                        pixel_values = torch.unsqueeze(pixel_values,0)
                        pixel_mask = torch.unsqueeze(pixel_mask,0)
                        ids = torch.unsqueeze(ids,0)
                        mask = torch.unsqueeze(mask,0)
                else:
                    # lxmert
                    normalized_boxes = torch.squeeze(batch['normalized_boxes']).to(device)
                    features = torch.squeeze(batch['features']).to(device)
                    if len(normalized_boxes.size())<3:
                        normalized_boxes = torch.unsqueeze(normalized_boxes,0)
                        features = torch.unsqueeze(features,0)
                        ids = torch.unsqueeze(ids,0)
                        mask = torch.unsqueeze(mask,0)
                   
                label=batch['labels'].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                if self.model_name == "vilt":
                    if self.use_tim_loss:
                        # TIM task
                        tim_ids, tim_mask, tim_token_type_ids, lbl_tim =self.prepare_itm_inputs(
                            ids,mask,token_type_ids
                            )
                        output, x_t, x_v, output_tim = self.model(
                            ids,mask,token_type_ids,pixel_values,pixel_mask,
                            tim_inputs=(tim_ids,tim_mask,tim_token_type_ids)
                            )
                    else:
                        output, x_t, x_v, _ = self.model(
                            ids,mask,token_type_ids,pixel_values,pixel_mask
                            )
                else:
                    # lxmert
                    if self.use_tim_loss:
                        tim_ids, tim_mask, tim_token_type_ids, lbl_tim = self.prepare_itm_inputs(
                            ids,mask,token_type_ids
                            )
                        output, x_t, x_v, output_tim = self.model(
                            ids,mask,token_type_ids,features,normalized_boxes,
                            tim_inputs=(tim_ids,tim_mask,tim_token_type_ids)
                            )
                    else:
                        output, x_t, x_v, _ = self.model(
                            ids,mask,token_type_ids,features,normalized_boxes
                            )
                
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
                    loss = (1-self.beta_itm) * loss_fn(output,label) + self.beta_itm * tim_loss_fn(output_tim,lbl_tim)                  
                elif self.use_loss_correction:
                    loss = loss_correction(T,loss_fn, output, label)
                else:
                    loss = loss_fn(output,label)
                # backward loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # predict
                # if not self.multilabel:
                #     pred = torch.argmax(self.softmax(output),dim=1)  
                #     target = torch.argmax(label,dim = 1)          
                #     num_correct = torch.sum(pred==target).item()
                #     num_samples = label.size(0)
                #     logger.info(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
                # else:
                #     pred = self.sigmoid(output)
                #     pred = torch.round(pred)
                    #num_correct = torch.sum(pred==label).item()
                    #num_samples = label.size(0)
                    #print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
            
            # predict val
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
        #return res_val

    def eval(self, dataloader, loss_fn, tim_loss_fn=None):
        eval_acc = []
        eval_loss = []
        predictions = []
        labels = []
        data_ids = []
        self.model.eval()

        for batch in tqdm(dataloader):
            ids = torch.squeeze(batch['input_ids']).to(device)
            mask = torch.squeeze(batch['attention_mask']).to(device)
            token_type_ids = torch.squeeze(batch['token_type_ids']).to(device)
            if self.model_name == "vilt":
                pixel_mask = torch.squeeze(batch['pixel_mask']).to(device)
                pixel_values = torch.squeeze(batch['pixel_values']).to(device)
            else:
                # lxmert
                normalized_boxes = torch.squeeze(batch['normalized_boxes']).to(device)
                features = torch.squeeze(batch['features']).to(device)
                if len(normalized_boxes.size())<3:
                    normalized_boxes = torch.unsqueeze(normalized_boxes,0)
                    features = torch.unsqueeze(features,0)
                    ids = torch.unsqueeze(ids,0)
                    mask = torch.unsqueeze(mask,0)
            label = batch['labels'].to(device)
            data_id = batch['data_id'].to(device)
            
            # Compute logits
            with torch.no_grad():
                if self.model_name == "vilt":
                    if self.use_tim_loss:
                        # TIM task
                        tim_ids, tim_mask,tim_token_type_ids, lbl_tim =self.prepare_itm_inputs(
                            ids,mask,token_type_ids
                            )
                        output, x_t, x_v, output_tim = self.model(
                            ids,mask,token_type_ids,pixel_values,pixel_mask,
                            tim_inputs=(tim_ids,tim_mask,tim_token_type_ids)
                            )
                    else:
                        output, x_t, x_v, _ = self.model(
                            ids,mask,token_type_ids,pixel_values,pixel_mask
                            )
                else:
                    # lxmert
                    if self.use_tim_loss:
                        tim_ids, tim_mask,tim_token_type_ids, lbl_tim = self.prepare_itm_inputs(
                            ids,mask,token_type_ids
                            )
                        output, x_t, x_v, output_tim = self.model(
                            ids,mask,token_type_ids,features,normalized_boxes,
                            tim_inputs=(tim_ids,tim_mask,tim_token_type_ids)
                            )
                    else:
                        output, x_t, x_v, _ = self.model(
                            ids,mask,token_type_ids,features,normalized_boxes
                            )
            
            # Compute loss
            label = label.type_as(output)
            # compute loss
            if self.use_clip_loss and self.use_tim_loss:
                logits_per_text = self.model.get_logits_per_text(x_t,x_v)
                loss = (1-(self.beta_itc+self.beta_itm)) * loss_fn(output,label) + self.beta_itc * clip_loss(logits_per_text) + self.beta_itm * tim_loss_fn(output_tim,lbl_tim)     
            elif self.use_clip_loss:
                logits_per_text = self.model.get_logits_per_text(x_t,x_v)
                loss = (1-self.beta_itc) * loss_fn(output,label) + self.beta_itc * clip_loss(logits_per_text) 
            elif self.use_tim_loss:
                loss = (1-self.beta_itm) * loss_fn(output,label) + self.beta_itm * tim_loss_fn(output_tim,lbl_tim) 
            elif self.use_loss_correction:
                loss = loss_correction(T,loss_fn, output, label)
            else:
                loss = loss_fn(output,label)

            eval_loss.append(loss.item())
            # Get the predictions
            if self.multilabel:
                soft_pred = self.sigmoid(output)
                pred = torch.round(soft_pred)
                target = label
            else:
                pred = torch.argmax(self.softmax(output), dim=1)  
                target = torch.argmax(label,dim = 1)  
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

        logger.info(f'test loss: {eval_loss:.4f} test acc: {(eval_acc):.4f}\n')
        
        y_pred = torch.stack(predictions)
        #print("y_pred",y_pred)
        y = torch.stack(labels)
        #print("y",y)
        data_ids = torch.stack(data_ids)
        
        res = {
            "data_id": data_ids,
            "loss": eval_loss,
            "predictions": y_pred,
            "labels": y
        }
        
        return res

