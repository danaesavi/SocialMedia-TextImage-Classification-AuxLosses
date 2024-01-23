import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import TxtOnly_Dataset
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from text_processing import Tweet_Preprocessing
from transformers import AutoTokenizer, AutoModel
import argparse
from config import *
from utils import prepare_data, prepare_text_data, agg_metrics_val, get_optimizer_params, loss_correction
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ------ LOGGING-----------------------------------------------------
import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S',
                   level=logging.INFO,
                   )
logger = logging.getLogger(__name__)
# ------ MODELS-----------------------------------------------------


class BERT(nn.Module):
    def __init__(self,model_dir, num_labels, dropout=0.1):
        super(BERT, self).__init__()
        self.bert_model = AutoModel.from_pretrained(model_dir)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(txt_feat_size, num_labels)
        
    def forward(self,ids,mask,token_type_ids):
        last_hidden, pooled_output= self.bert_model(ids,attention_mask=mask,token_type_ids=token_type_ids, return_dict=False)
        dropout_output = self.dropout(last_hidden[:,0,:])
        #dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        return linear_output
   
class BERNICE(nn.Module):
    def __init__(self,model_dir, num_labels, dropout=0.1):
        super().__init__()
        self.bert_model = AutoModel.from_pretrained(model_dir)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(txt_feat_size, num_labels)
        
    def forward(self,ids,mask):
        last_hidden, pooled_output = self.bert_model(ids,attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(last_hidden[:,0,:])
        # dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        return linear_output

class RoBERTa(nn.Module):
    def __init__(self,model_dir, num_labels, dropout=0.1):
        super(RoBERTa, self).__init__()
        self.bert_model = AutoModel.from_pretrained(model_dir)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(txt_feat_size, num_labels)
        
    def forward(self,ids,mask):
        _,pooled_output= self.bert_model(ids,attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(pooled_output)
        return linear_output
   
class TextModel(object):
    """
    TextModel class
    """
    def __init__(self, config, model_name, freeze = False):
        """ Initialization """
        self.batch_size = config.batch_size
        self.num_labels = config.num_labels
        self.model_name = model_name
        self.model_dir = MODEL_DIR_DICT[self.model_name]
        self.max_length = config.max_length
        self.dropout = config.dropout
        self.use_loss_correction = config.use_loss_correction
      
        # tokenizer
        if self.model_name == "bernice":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, model_max_length=self.max_length) 
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir) 
        
        # model
        if self.model_name == "roberta":
            model = RoBERTa(self.model_dir, self.num_labels, dropout=self.dropout)
        elif self.model_name == "bernice":
            self.model = BERNICE(self.model_dir, self.num_labels, dropout=self.dropout)
        else:
            self.model = BERT(self.model_dir, self.num_labels, dropout=self.dropout)
        
        if freeze:
            for param in self.model.bert_model.parameters():
                param.requires_grad = False
        
        self.model.to(device)
        print(self.model)

        self.softmax = nn.Softmax(dim=1)
    
    def load_data(self,data, testing=False ,eval_txt_test=False, task_name=None):

        train, y_vector_tr, val, y_vector_val, test, y_vector_te, class_weights, image_adds = prepare_data(data, self.num_labels, testing=testing)
        tr_dataset = TxtOnly_Dataset(self.model_name, train.tweet_id.values,train.text.values,y_vector_tr,self.tokenizer, self.max_length,task_name)
        train_loader = DataLoader(tr_dataset, batch_size=self.batch_size,shuffle=True)
        val_dataset = TxtOnly_Dataset(self.model_name, val.tweet_id.values, val.text.values,y_vector_val,self.tokenizer, self.max_length,task_name)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,shuffle=False)
        te_dataset = TxtOnly_Dataset(self.model_name, test.tweet_id.values, test.text.values,y_vector_te,self.tokenizer, self.max_length,task_name)
        test_loader = DataLoader(te_dataset, batch_size=self.batch_size,shuffle=False)
        if eval_txt_test:
            # text_only
            txt_test, y_txt_te = prepare_text_data(num_labels=self.num_labels, testing=testing)
            txt_te_dataset = TxtOnly_Dataset(self.model_name, txt_test.tweet_id.values, txt_test.text.values, y_txt_te, self.tokenizer, self.max_length, task_name)
            txt_te_loader = DataLoader(txt_te_dataset, batch_size=self.batch_size,shuffle=False) 
        else:
            txt_te_loader= None
        return train_loader, val_loader, test_loader, class_weights, txt_te_loader
    

    def train(self,dataloader,val_dataloader,epochs,loss_fn,lr,weight_decay,
    te_dataloader=None,model_path=None,val_filename=None,te_filename=None):  

        #Initialize Optimizer
        named_parameters = self.model.named_parameters()
        optimizer_params = get_optimizer_params(named_parameters, weight_decay, lr)
        optimizer = optim.AdamW(optimizer_params, lr=lr)
        #optimizer= optim.Adam(self.model.parameters(), lr= lr, weight_decay = weight_decay)
        
        self.model.train()
        res_val, res_te = [], []
        for  epoch in range(epochs):
            print(epoch)
            loop=tqdm(enumerate(dataloader),leave=False,total=len(dataloader))
            for batch, dl in loop:
                ids=dl['ids'].to(device)
                mask= dl['mask'].to(device)
                label=dl['target'].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                if self.model_name not in {"roberta", "bernice"}:
                    token_type_ids=dl['token_type_ids'].to(device)
                    output=self.model(
                        ids=ids,
                        mask=mask,
                        token_type_ids=token_type_ids)
                else:
                    # roberta, bernice
                    output=self.model(
                        ids=ids,
                        mask=mask)
                label = label.type_as(output)
                # compute loss
                if self.use_loss_correction:
                    loss = loss_correction(T,loss_fn, output, label)
                else:
                    loss=loss_fn(output,label)
                # backward loss
                loss.backward()
                optimizer.step()
                # predict
                pred = torch.argmax(self.softmax(output),dim=1)  
                target = torch.argmax(label,dim = 1)          
                num_correct = torch.sum(pred==target).item()
                num_samples = label.size(0)
                accuracy = num_correct/num_samples
                print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
                
                # Show progress while training
                loop.set_description(f'Epoch={epoch}/{epochs}')
                loop.set_postfix(loss=loss.item(),acc=accuracy)
            
            # predict val
            print("val")
            res_val_d = self.eval(val_dataloader,loss_fn)
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
                res_te_d = self.eval(te_dataloader,loss_fn)
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

    def eval(self, dataloader, loss_fn):
        eval_acc = []
        eval_loss = []
        predictions = []
        labels = []
        data_ids = []
        loop=tqdm(enumerate(dataloader),leave=False,total=len(dataloader))
        self.model.eval()
        
        for batch, dl in loop:
            ids = dl['ids'].to(device)                
            mask = dl['mask'].to(device)
            label = dl['target'].to(device) 
            data_id = dl['data_id'].to(device)
            if self.model_name not in {"roberta", "bernice"}:
                token_type_ids=dl['token_type_ids'].to(device)
            # Compute logits
            with torch.no_grad():
                if self.model_name not in {"roberta","bernice"}:
                    output=self.model(
                        ids=ids,
                        mask=mask,
                        token_type_ids=token_type_ids)
                else:
                    # roberta, bernice
                    output=self.model(
                        ids=ids,
                        mask=mask,
                        )
            label = label.type_as(output)
            # Compute loss
            if self.use_loss_correction:
                loss = loss_correction(T,loss_fn, output, label)
            else:
                loss=loss_fn(output,label)
            eval_loss.append(loss.item())
            # Get the predictions
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

        print(f'loss: {eval_loss:.4f} acc: {(eval_acc):.4f}\n')
        
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




        










