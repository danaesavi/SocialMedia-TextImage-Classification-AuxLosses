import sys
import torch
import torch.nn as nn
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    AutoImageProcessor,
    ViTFeatureExtractor,
    AutoTokenizer,
    AutoModel
)
import pandas as pd
from utils import (
    get_conv_model, 
    prepare_data,
     prepare_text_data,
    get_optimizer_params,
    agg_metrics_val,
    clip_loss,
    loss_correction
)
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from datasets import MM_Dataset, MM_CNN_Dataset
from torch.utils.data import DataLoader
from config import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ------ LOGGING-----------------------------------------------------
import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S',
                   level=logging.INFO,
                   )
logger = logging.getLogger(__name__)

# ------- MODELS ------------------------------------------------------

def get_fusion_model(fusion_name):
     FUSION_DICT = {
        'xatt': XATT,
        'concat_cnn': CNNImgConcat,
     }
     return FUSION_DICT[fusion_name]

################## VisionTextDualEncoder #########################
class MM_Model(nn.Module):
    def __init__(self, num_labels, txt_model_name, img_model_name, dropout, fusion_name='concat'):
        super(MM_Model, self).__init__()
        self.num_labels = num_labels
        self.fusion_name = fusion_name
        self.txt_model_name = txt_model_name
        self.img_model_name = img_model_name
        txt_model_dir = MODEL_DIR_DICT[self.txt_model_name]
        img_model_dir = MODEL_DIR_DICT[self.img_model_name]
        self.dual_encoder = VisionTextDualEncoderModel.from_vision_text_pretrained(
                img_model_dir, txt_model_dir
            ).to(device)
        

        #for param in self.dual_encoder.parameters():
        #    param.requires_grad = False
        # freeze vision 
        for name, param in self.dual_encoder.named_parameters():
            if 'vision' in name:
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)

        self.fc_Q = nn.Linear(txt_feat_size, fixed_feat_size)
        self.fc_K = nn.Linear(img_feat_size, fixed_feat_size)
        self.fc_V = nn.Linear(img_feat_size, fixed_feat_size)
        self.attention = Scaled_Dot_Product_Attention()
        
        self.aspectattention = nn.Linear(fixed_feat_size, 1)
        self.m = nn.Softmax(dim=1)

        self.linear_fusion = nn.Linear(fixed_feat_size * 2, fixed_feat_size)
        self.relu = nn.ReLU()
        self.linear_cls = nn.Linear(fixed_feat_size, self.num_labels)
        self.linear_tim = nn.Linear(fixed_feat_size, 2)
        self.linear_iadds = nn.Linear(fixed_feat_size, 2)
        self.z = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.linear_gmu_t = nn.Linear(fixed_feat_size, 2*fixed_feat_size)
        self.linear_gmu_v = nn.Linear(fixed_feat_size, 2*fixed_feat_size)
        
    def mm_fusion(self,x_t, x_v, x_v_pool=None, x_t_pool=None):
        if self.fusion_name=="concat":
            # Concatenate text and image features using cls representation
            xt_xv = torch.cat((x_t[:, 0, :], x_v[:, 0, :]), dim=1)
            xt_xv = self.relu(self.linear_fusion(xt_xv))
            return xt_xv
        
        if self.fusion_name=="attention":
            print("attention")
            #x_t: 16 X 128 X 768
            #x_v: 16 X 197 X 768
            
            # QXK^t: 128X768 197X768 --> 128X197 (attention scores)
            # 128X197  197X768  -->  128X768 (output)

            N, L, E = x_t.size()
            Q, K, V  = self.fc_Q(x_t), self.fc_K(x_v), self.fc_V(x_v)
            scale = K.size(-1) ** -0.5
            x_v, attn_output_weight = self.attention(Q, K, V, scale)
            x_v = x_v.view(N, L, E)
            xt_xv = torch.cat((x_t[:, 0, :], x_v[:, 0, :]), dim=1)
            xt_xv = self.relu(self.linear_fusion(xt_xv))
            return xt_xv

        if self.fusion_name=="aspect-att":
            print("aspect-att")
            #x_t: 16 X 128 X 768
            #x_v: 16 X 197 X 768
            N, L, EM = x_t.size()
            V = torch.stack((x_t_pool, x_v_pool), dim=0) # 2x768
            V = torch.reshape(V, (N,2,EM))
            #print("V",V.size())
            E = self.tanh(self.aspectattention(V)) # 2x1
            #print("E",E.size())
            attn_weights = self.m(E) 
            attn_weights = torch.transpose(attn_weights,1,2) # 1x2
            #print("attn_weight",attn_weights.size())
            xt_xv = torch.matmul(attn_weights, V).squeeze(1) # 768x1
            #print("attnout",xt_xv.size())
            xt_xv = self.relu(xt_xv)
            return xt_xv
        
        if self.fusion_name=="gmu":
            print("gmu")
            #x_t: 16 X 128 X 768
            #x_v: 16 X 197 X 768
            # gmu fusion
            x_v_prime = self.linear_gmu_v(x_v[:, 0, :])
            x_t_prime = self.linear_gmu_t(x_t[:, 0, :])
            xt_cat_xv = torch.cat((x_t[:, 0, :],x_v[:, 0, :]), dim=1)
            z = self.z(xt_cat_xv)
            xt_xv = z * x_t_prime + (1 - z) * x_v_prime 
            xt_xv = self.relu(self.linear_fusion(xt_xv))  
            return xt_xv
        

        
    def forward(self,ids,mask,pixel_values,tim_inputs=None,iadds_task=False):
        outputs = self.dual_encoder(
                input_ids=ids,
                attention_mask=mask,
                pixel_values=pixel_values,
                return_loss=True,
            )
        xv_last_hidden = outputs.vision_model_output.last_hidden_state # BXLXE
        x_v_pool = outputs.vision_model_output.pooler_output # BXE
        xt_last_hidden = outputs.text_model_output.last_hidden_state # BXLXE
        x_t_pool = outputs.text_model_output.pooler_output # BXE
        logits_per_text = outputs.logits_per_text # this is the text-image similarity score
        xt_xv = self.mm_fusion(xt_last_hidden,xv_last_hidden,
        x_v_pool=x_v_pool,x_t_pool=x_t_pool)
        mm_features = xt_xv
        xt_xv = self.dropout(xt_xv)
        out_cls = self.linear_cls(xt_xv)
        

        # TIM
        if tim_inputs is not None:
            tim_ids, tim_mask = tim_inputs
            tim_outputs = self.dual_encoder(
                input_ids=tim_ids,
                attention_mask=tim_mask,
                pixel_values=pixel_values,
                return_loss=True,
            )
            xv_last_hidden_tim = tim_outputs.vision_model_output.last_hidden_state # BXLXE
            #x_v_pool = outputs.vision_model_output.pooler_output # BXE
            
            xt_last_hidden_tim = tim_outputs.text_model_output.last_hidden_state # BXLXE
            #x_t_pool = outputs.text_model_output.pooler_output # BXE
            xt_xv_tim = self.mm_fusion(xt_last_hidden_tim,xv_last_hidden_tim)
            out_tim = self.linear_tim(xt_xv_tim)
        else:
            out_tim = None
        
        # IADDS
        if iadds_task:
            out_iadds = self.linear_iadds(xt_xv)
        else:
            out_iadds = None

        
        return out_cls, logits_per_text, out_tim, out_iadds, mm_features
    
class Scaled_Dot_Product_Attention(nn.Module):
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        # compute attention scores using query and key
        attention = torch.matmul(Q, K.permute(0, 2, 1)) 
        attention_scores = attention
        if scale:
            attention = attention * scale

        # get attention weights
        attention = F.softmax(attention, dim=-1)
        # compute the weighted sum of the value head
        context = torch.matmul(attention, V)
        return context, attention_scores
    
############## CNN #####################

class Self_Attn2(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(Self_Attn2,self).__init__()
        self.chanel_in = in_dim        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) 

        self.linear_proj_1 = nn.Linear(256,fixed_feat_size) # 200
        self.linear_proj_2 = nn.Linear(2048,fixed_feat_size) # 2048,200
        self.linear_proj_3 = nn.Linear(txt_feat_size,fixed_feat_size) # 768, 200
        self.att = nn.MultiheadAttention(fixed_feat_size, 1, batch_first=True) # 200
    def forward(self,last_hidden, last_conv):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = last_conv.size()
        C = 200
     
        # text
        proj_query = self.linear_proj_3(last_hidden)

        proj_key =  self.key_conv(last_conv).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        proj_key =  self.linear_proj_1(proj_key.permute(0,2,1)) # B X C=FIXED X (W*H)
        
        proj_value = self.value_conv(last_conv).view(m_batchsize,-1,width*height) # B X C X N
        proj_value =  self.linear_proj_2(proj_value.permute(0,2,1))  # B X C=FIXED X (W*H)
    
        out, attention = self.att(proj_query,proj_key,proj_value)
        

        return out, attention

class Self_Attn1(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(Self_Attn1,self).__init__()
        self.chanel_in = in_dim        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) 

        self.linear_proj_1 = nn.Linear(256,fixed_feat_size) # 200
        self.linear_proj_2 = nn.Linear(img_feat_size_cnn,fixed_feat_size) # 2048, 200
        self.linear_proj_3 = nn.Linear(txt_feat_size,fixed_feat_size) # 768,200
        self.att = nn.MultiheadAttention(fixed_feat_size, 1, batch_first=True) # 200
    def forward(self,last_hidden, last_conv):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        print("att img")
        m_batchsize,C,width ,height = last_conv.size()
        C = 768 # 200
        proj_query  = self.query_conv(last_conv).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_query = self.linear_proj_1(proj_query)
        # text
        proj_key = self.linear_proj_3(last_hidden)
        # text
        proj_value = self.linear_proj_3(last_hidden)
    
        out, attention = self.att(proj_query,proj_key,proj_value)

        out = out.permute(0,2,1)
        out = out.view(m_batchsize,C,width,height)

        return out, attention



class MMLate_Model(object):
    """
    MMLate_Model class
    """       
    
    def __init__(self, config, txt_model_name, img_model_name, fusion_name, multilabel=False):
        """ Initialization """
        self.batch_size = config.batch_size
        self.num_labels = config.num_labels
        self.multilabel = multilabel
        self.use_clip_loss = config.use_clip_loss
        self.beta_itc = config.beta_itc
        self.use_tim_loss = config.use_tim_loss
        self.beta_itm = config.beta_itm
        self.use_iadds_loss = config.use_iadds_loss
        self.beta_iadds = config.beta_iadds
        self.use_loss_correction = config.use_loss_correction
        self.txt_model_name = txt_model_name
        self.img_model_name = img_model_name
        txt_model_dir = MODEL_DIR_DICT[self.txt_model_name]
        img_model_dir = MODEL_DIR_DICT[self.img_model_name]
        self.max_length = config.max_length
        self.cnn = self.img_model_name in {"resnet50","resnet152"}
        self.tokenizer = AutoTokenizer.from_pretrained(txt_model_dir) 
        if not self.cnn:
            #image_processor = AutoImageProcessor.from_pretrained(img_model_dir)
            feature_extractor = ViTFeatureExtractor.from_pretrained(img_model_dir)
            self.processor = VisionTextDualEncoderProcessor(feature_extractor, self.tokenizer)
            self.model = MM_Model(
                self.num_labels, 
                self.txt_model_name, 
                self.img_model_name, 
                config.dropout, 
                fusion_name=fusion_name
                )

        else:
            self.model = get_fusion_model(fusion_name)(self.num_labels,
                                                        self.txt_model_name, 
                                                        self.img_model_name,
                                                        config.dropout)
            
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
           
    def load_saved_model(self,model_path):
        self.model.load_state_dict(torch.load(model_path)
                                   )
    def load_data(self, data, img_file_fmt, testing=False, nsamples=-1, saved_features=False, task_name=None,
                  eval_txt_test=False, compute_class_weights=True, random_labels=False):
        
        train, y_vector_tr, val, y_vector_val, test, y_vector_te, class_weights, image_adds = prepare_data(
            data, self.num_labels, testing=testing, nsamples=nsamples, compute_class_weights=compute_class_weights,
            random_labels=random_labels, load_image_adds=self.use_iadds_loss, multilabel=self.multilabel)
        
        if self.cnn:
            tr_dataset = MM_CNN_Dataset(self.txt_model_name, train.tweet_id.values,train.text.values,y_vector_tr,self.tokenizer, self.max_length, img_file_fmt)
            val_dataset = MM_CNN_Dataset(self.txt_model_name, val.tweet_id.values,val.text.values,y_vector_val,self.tokenizer, self.max_length, img_file_fmt)
            te_dataset = MM_CNN_Dataset(self.txt_model_name, test.tweet_id.values,test.text.values,y_vector_te,self.tokenizer, self.max_length, img_file_fmt)
        else:
            
            train_ids = train.tweet_id.values
            val_ids = val.tweet_id.values
            test_ids = test.tweet_id.values

            tr_dataset = MM_Dataset(
                train_ids,train.text.values,y_vector_tr,self.processor, self.max_length,
              img_file_fmt=img_file_fmt, saved_features=saved_features, task_name=task_name, image_adds=image_adds["train"])
            val_dataset = MM_Dataset(
                val_ids,val.text.values,y_vector_val,self.processor, self.max_length, 
             img_file_fmt=img_file_fmt, saved_features=saved_features, task_name=task_name, image_adds=image_adds["val"])
            te_dataset = MM_Dataset(
                test_ids,test.text.values,y_vector_te,self.processor, self.max_length, 
             img_file_fmt=img_file_fmt, saved_features=saved_features, task_name=task_name, image_adds=image_adds["test"])
            if eval_txt_test:
                txt_test, y_txt_te, image_adds = prepare_text_data(num_labels=self.num_labels, testing=testing,
                                                       load_image_adds=self.use_iadds_loss)
                txt_te_dataset = MM_Dataset(
                    txt_test.tweet_id.values,txt_test.text.values, y_txt_te, self.processor, 
                    self.max_length, empty_image=EMPTY_IMG, saved_features=saved_features, task_name=task_name,
                    image_adds=image_adds)
                txt_te_loader = DataLoader(txt_te_dataset, batch_size=self.batch_size,shuffle=False) 

            else:
                txt_te_loader= None
        
        train_loader = DataLoader(tr_dataset, batch_size=self.batch_size, shuffle=True) 
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,shuffle=False)        
        test_loader = DataLoader(te_dataset, batch_size=self.batch_size,shuffle=False)
        return train_loader, val_loader ,test_loader, class_weights, txt_te_loader
    
    def prepare_itm_inputs(self, ids, mask):
        # replace text with 0.5 probability
        tim_ids = ids.clone().detach()
        tim_mask = mask.clone().detach()
        labels_tim = []
        m_batchsize, _ = ids.size()
        if m_batchsize>1:
            for idx in range(m_batchsize):
                change_text = np.random.choice([True, False])
                if change_text:
                    # mismatch
                    labels_tim.append(0) 
                    indexes = set(range(m_batchsize)) - {idx}
                    new_idx = np.random.choice(list(indexes))
                    tim_ids[idx] = ids[new_idx]
                    tim_mask[idx] = mask[new_idx]
                else:
                    # match
                    labels_tim.append(1)
        else:
            labels_tim.append(1)
        tim_ids = tim_ids.to(device)
        tim_mask = tim_mask.to(device)
        lbl_tim = torch.tensor(labels_tim, dtype=torch.long).to(device)

        return tim_ids, tim_mask, lbl_tim

    def train(self,dataloader,val_dataloader,epochs,loss_fn,lr,weight_decay,
    tim_loss_fn = None,iadds_loss_fn=None,te_dataloader=None,model_path=None,val_filename=None,
    te_filename=None):
        #Initialize Optimizer
        named_parameters = self.model.named_parameters()
        optimizer_params = get_optimizer_params(named_parameters, weight_decay, lr)
        optimizer = optim.AdamW(optimizer_params, lr=lr)
        print("model parameters",sum(p.numel() for p in self.model.parameters()))

        #optimizer= optim.Adam(self.model.parameters(), lr= lr, weight_decay = weight_decay)
        res_val, res_te = [], []
        for epoch in range(epochs):
            self.model.train()
            print("Epoch:",epoch+1)
            for batch in tqdm(dataloader):
                if self.cnn:
                    ids = batch['ids'].to(device)
                    mask = batch['mask'].to(device)
                    token_type_ids =batch['token_type_ids'].to(device)
                    pixel_values = batch['pixel_values'].to(device)
                   
                else:
                    ids = torch.squeeze(batch['input_ids'])
                    mask = torch.squeeze(batch['attention_mask'])
                    pixel_values = torch.squeeze(batch['pixel_values'])
                    if len(pixel_values.size()) < 4:
                        pixel_values = torch.unsqueeze(pixel_values, 0)
                        ids = torch.unsqueeze(ids,0)
                        mask = torch.unsqueeze(mask,0)
                    pixel_values = pixel_values.to(device)
                    ids = ids.to(device)
                    mask = mask.to(device)
                
                
                label=batch['labels'].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                if self.cnn:
                    output, x_t, x_v = self.model(ids,mask,token_type_ids,pixel_values)
                else:
                    if self.use_tim_loss:
                        # TIM task
                        tim_ids, tim_mask, lbl_tim = self.prepare_itm_inputs(ids,mask)
                        tim_inputs= (tim_ids,tim_mask)
                    else:
                        tim_inputs= None
                   
                    output, logits_per_text, output_tim, output_iadds, _ = self.model(
                            ids,mask,pixel_values,
                            tim_inputs=tim_inputs,
                            iadds_task=self.use_iadds_loss
                            )
                  

                label = label.type_as(output)
                # compute loss
                if self.use_clip_loss and self.use_tim_loss:
                    loss = (1-(self.beta_itc+self.beta_itm)) * loss_fn(output,label) + self.beta_itc * clip_loss(logits_per_text) + self.beta_itm * tim_loss_fn(output_tim,lbl_tim)
                elif self.use_clip_loss:
                    loss = (1-self.beta_itc) * loss_fn(output,label) + self.beta_itc * clip_loss(logits_per_text)
                elif self.use_tim_loss:
                    loss = (1-self.beta_itm) * loss_fn(output,label) + self.beta_itm * tim_loss_fn(output_tim,lbl_tim)
                elif self.use_iadds_loss:
                    iadds_label = batch["image_adds"].to(device)
                    #iadds_label = iadds_label.type_as(output_iadds)
                    print("img_adds", iadds_label)
                    loss = (1-self.beta_iadds) * loss_fn(output,label) + self.beta_iadds * iadds_loss_fn(output_iadds,iadds_label)
                elif self.use_loss_correction:
                    loss = loss_correction(T,loss_fn, output, label)
                else:
                    loss = loss_fn(output,label)
                # backward loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # predict train
                if not self.multilabel:
                    pred = torch.argmax(self.softmax(output),dim=1)  
                    target = torch.argmax(label,dim = 1)          
                    num_correct = torch.sum(pred==target).item()
                    num_samples = label.size(0)
                    print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
                else:
                    pred = self.sigmoid(output)
                    pred = torch.round(pred)
                    num_correct = torch.sum(pred==label).item()
                    num_samples = label.size(0)
                    print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
                
            # predict val
            print("val")
            res_val_d = self.eval(val_dataloader,loss_fn,tim_loss_fn=tim_loss_fn,iadds_loss_fn=iadds_loss_fn)
            res_val_d["epoch"] = epoch
            res_val.append(res_val_d)
            if val_filename != None and (epoch%2 == 0 or epoch==epochs-1):
                logger.info("Compute metrics (val)")
                metrics_val = agg_metrics_val(res_val, metric_names, self.num_labels)
                pd.DataFrame(metrics_val).to_csv(val_filename,index=False)
                logger.info("{} saved!".format(val_filename))

            if te_dataloader != None:
                # predict test
                print("test")
                res_te_d = self.eval(te_dataloader,loss_fn,tim_loss_fn=tim_loss_fn,iadds_loss_fn=iadds_loss_fn)
                res_te_d["epoch"] = epoch
                res_te.append(res_te_d)
                if te_filename != None and (epoch%2 == 0 or epoch==epochs-1):
                    logger.info("Compute metrics (test)")
                    metrics_te = agg_metrics_val(res_te, metric_names, self.num_labels)
                    pd.DataFrame(metrics_te).to_csv(te_filename,index=False)
                    logger.info("{} saved!".format(te_filename))

        if model_path != None:
            torch.save(self.model.state_dict(), model_path)
            logger.info("{} saved".format(model_path))
        #return res_val, res_te
    
    def eval(self, dataloader, loss_fn, tim_loss_fn = None,iadds_loss_fn=None):
        eval_acc = []
        eval_loss = []
        soft_preds_0 = []
        soft_preds_1 = []
        predictions = []
        labels = []
        data_ids = []
        self.model.eval()
        
        for batch in tqdm(dataloader):
            if self.cnn:
                ids = batch['ids'].to(device)
                mask = batch['mask'].to(device)
                token_type_ids =batch['token_type_ids'].to(device)
                pixel_values = batch['pixel_values'].to(device)
            else:
                ids = torch.squeeze(batch['input_ids']).to(device)
                mask = torch.squeeze(batch['attention_mask']).to(device)
                pixel_values = torch.squeeze(batch['pixel_values'])
                pixel_values = pixel_values.to(device)
                #print("pixel values",pixel_values.size())
            label=batch['labels'].to(device)
            data_id = batch['data_id'].to(device)
            
            # Compute logits
            with torch.no_grad():
                if self.cnn:
                    output, x_t, x_v = self.model(ids,mask,token_type_ids,pixel_values)
                    #similarity_score = compute_batch_dot_product(x_t,x_v)
                else:
                    if self.use_tim_loss:
                        # TIM task
                        tim_ids, tim_mask, lbl_tim =self.prepare_itm_inputs(ids,mask)
                        tim_inputs = (tim_ids,tim_mask)
                    else:
                        tim_inputs = None
                    # forward
                    output, logits_per_text, output_tim, output_iadds, _ = self.model(
                        ids,mask,pixel_values,
                        tim_inputs=tim_inputs,
                        iadds_task=self.use_iadds_loss
                        )
                 
                        
            # Compute loss
            label = label.type_as(output)
            if self.use_clip_loss and self.use_tim_loss:
                loss = (1-(self.beta_itc+self.beta_itm)) * loss_fn(output,label) + self.beta_itc * clip_loss(logits_per_text) + self.beta_itm * tim_loss_fn(output_tim,lbl_tim)
            elif self.use_clip_loss:
                loss = (1-self.beta_itc) * loss_fn(output,label) + self.beta_itc * clip_loss(logits_per_text)
            elif self.use_tim_loss:
                loss = (1-self.beta_itm) * loss_fn(output,label) + self.beta_itm * tim_loss_fn(output_tim,lbl_tim)
            elif self.use_iadds_loss:
                iadds_label = batch["image_adds"].to(device)
                loss = (1-self.beta_iadds) * loss_fn(output,label) + self.beta_iadds * iadds_loss_fn(output_iadds,iadds_label)
            elif self.use_loss_correction:
                loss = loss_correction(T,loss_fn, output, label)
            else:
                loss = loss_fn(output,label)
            eval_loss.append(loss.item())
            # Get the predictions
            if not self.multilabel:
                soft_pred = self.softmax(output)
                #soft_pred_0 = soft_pred[:,0]
                #soft_pred_1 = soft_pred[:,1]
                pred = torch.argmax(soft_pred, dim=1)
                target = torch.argmax(label,dim = 1)  
            else:
                soft_pred = self.sigmoid(output)
                pred = torch.round(soft_pred)
                target = label
            # Calculate the accuracy rate
            accuracy = (pred == target).cpu().numpy().mean() * 100
            eval_acc.append(accuracy)
          
            predictions += pred
            labels += target
            data_ids += data_id
        
        # Compute the average accuracy and loss over the validation set.
        eval_loss = np.mean(eval_loss)
        eval_acc = np.mean(eval_acc)

        print(f'loss: {eval_loss:.4f} acc: {(eval_acc):.4f}\n')
        #y_soft_pred_0 = torch.stack(soft_preds_0)
        #y_soft_pred_1 = torch.stack(soft_preds_1)
        y_pred = torch.stack(predictions)
        #print("y_pred_0",y_soft_pred_0)
        #print("y_pred_1",y_soft_pred_0)

        y = torch.stack(labels)
        #print("y",y)
        data_ids = torch.stack(data_ids)
        
        res = {
            "data_id": data_ids,
            "loss": eval_loss,
            "predictions": y_pred,
            #"soft_pred_0": y_soft_pred_0,
            #"soft_pred_1": y_soft_pred_1,
            "labels": y
        }
        
        return res
    
    def compute_predictions(self, dataloader):
        soft_preds_0 = []
        soft_preds_1 = []
        predictions = []
        data_ids = []
        self.model.eval()
        
        for batch in tqdm(dataloader):
            if self.cnn:
                ids = batch['ids'].to(device)
                mask = batch['mask'].to(device)
                token_type_ids =batch['token_type_ids'].to(device)
                pixel_values = batch['pixel_values'].to(device)
            else:
                ids = torch.squeeze(batch['input_ids']).to(device)
                mask = torch.squeeze(batch['attention_mask']).to(device)
                pixel_values = torch.squeeze(batch['pixel_values'])
                pixel_values = pixel_values.to(device)
                #print("pixel values",pixel_values.size())
            
            data_id = batch['data_id'].to(device)
            
            # Compute logits
            with torch.no_grad():
                if self.cnn:
                    output, x_t, x_v = self.model(ids,mask,token_type_ids,pixel_values)
                else:
                    if self.use_tim_loss:
                        # TIM task
                        tim_ids, tim_mask, lbl_tim =self.prepare_itm_inputs(ids,mask)
                        tim_inputs = (tim_ids,tim_mask)
                    else:
                        tim_inputs = None
                    # forward
                    output, logits_per_text, output_tim, output_iadds = self.model(
                        ids,mask,pixel_values,
                        tim_inputs=tim_inputs,
                        iadds_task=self.use_iadds_loss
                        )
            
            # Get the predictions
            if self.multilabel:
                soft_pred = self.sigmoid(output)
                pred = torch.round(soft_pred)
            else:
                soft_pred = self.softmax(output)
                pred = torch.argmax(soft_pred, dim=1)
           
            predictions += pred
            data_ids += data_id
        
      
        y_pred = torch.stack(predictions)
        data_ids = torch.stack(data_ids)
        
        res = {
            "data_id": data_ids,
            "predictions": y_pred,
          
        }
        
        return res
    
    def extract_features(self, dataloader):
        self.model.eval()
        features = []
        labels = []
        for batch in tqdm(dataloader):
            ids = torch.squeeze(batch['input_ids']).to(device)
            mask = torch.squeeze(batch['attention_mask']).to(device)
            pixel_values = torch.squeeze(batch['pixel_values'])
            pixel_values = pixel_values.to(device)            
            data_id = batch['data_id'].to(device)
            label=batch['labels'].to(device)
            
            # Compute logits
            with torch.no_grad():
                if self.use_tim_loss:
                    # TIM task
                    tim_ids, tim_mask, lbl_tim =self.prepare_itm_inputs(ids,mask)
                    tim_inputs = (tim_ids,tim_mask)
                else:
                    tim_inputs = None
                # forward
                output, logits_per_text, output_tim, output_iadds, mm_feats = self.model(
                    ids,mask,pixel_values,
                    tim_inputs=tim_inputs,
                    iadds_task=self.use_iadds_loss
                    )
            features += mm_feats
            target = torch.argmax(label,dim = 1)  
            labels += target
            print(len(labels))
                
        y = torch.stack(labels)
        feats = torch.stack(features)
        print("feats", feats.size())
        print("y", y.size())
        
        return feats, y
       

