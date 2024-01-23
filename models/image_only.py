import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datasets import ImgOnly_Dataset, ImgOnlyCNN_Dataset
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel
import argparse
from sklearn.utils.class_weight import compute_class_weight
from torchmetrics.classification import BinaryF1Score
from PIL import Image
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from config import *
from utils import get_conv_model, prepare_data, agg_metrics_val, get_optimizer_params
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ------ LOGGING-----------------------------------------------------
import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S',
                   level=logging.INFO,
                   )
logger = logging.getLogger(__name__)
# ------ MODELS------------------------------------------------------

class BEiT(nn.Module):
    def __init__(self, model_dir, num_labels, dropout=0.1):
        super(BEiT, self).__init__()
        self.model = AutoModel.from_pretrained(model_dir)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(img_feat_size, num_labels)
        
    def forward(self,pixel_values):
        _,pooled_output= self.model(pixel_values, return_dict=False)
        linear_output = self.linear(pooled_output)
        return linear_output

class DEiT(nn.Module):
    def __init__(self, model_dir, num_labels, dropout=0.1):
        super(DEiT, self).__init__()
        self.model = AutoModel.from_pretrained(model_dir)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(img_feat_size, num_labels)
        
    def forward(self,pixel_values):
        _,pooled_output = self.model(pixel_values, return_dict=False)
        linear_output = self.linear(pooled_output)
        return linear_output

class CNN(nn.Module):
    def __init__(self, model_dir, model_name, num_labels, feature_extract = False):
        super(CNN, self).__init__()
        self.model_name = model_name
        self.feature_extract = feature_extract
        self.net = get_conv_model(self.model_name)
        self.net.load_state_dict(torch.load(model_dir))
        self.set_parameter_requires_grad()
        num_ftrs = self.net.fc.in_features # 2048
        self.net.fc = nn.Linear(num_ftrs, num_labels)
        # params to optimize
        print("Params to learn:")
        params_to_update = self.net.parameters()
        if self.feature_extract:
            params_to_update = []
            for name,param in self.net.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)
        else:
            for name,param in self.net.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)
        self.params_to_update = params_to_update
        self.net = self.net.to(device)
        
    
    def set_parameter_requires_grad(self):
        if self.feature_extract:
            for param in self.net.parameters():
                param.requires_grad = False

    def forward(self,x):
        x_out = self.net(x)
        return x_out

class CNNAtt(nn.Module):
    def __init__(self, model_dir, model_name, num_labels, output_layer="layer4", feature_extract = False):
        super(CNNAtt, self).__init__()
        self.output_layer = output_layer
        self.feature_extract = feature_extract
        self.pretrained = get_conv_model(self.model_name)
        self.pretrained.load_state_dict(torch.load(model_dir))
        self.children_list = []
        for n,c in self.pretrained.named_children():
            self.children_list.append(c)
            if n == self.output_layer:
                break
        self.net = nn.Sequential(*self.children_list)
        self.net = self.net.to(device)
        self.pretrained = None
        self.set_parameter_requires_grad()        
        self.att = Self_Attn(img_feat_size_cnn) # 2048
        self.avg_pool = nn.AvgPool2d(7)
        self.num_ftrs = img_feat_size_cnn # 2048
        self.num_labels = num_labels
        self.linear = nn.Linear(self.num_ftrs, self.num_labels)
        params_to_update = self.net.parameters()
        print("Params to learn:")
        if self.feature_extract:
            params_to_update = []
            for name,param in self.net.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)
        else:
            for name,param in self.net.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)
        self.params_to_update = params_to_update

    def set_parameter_requires_grad(self):
        if self.feature_extract:
            for param in self.net.parameters():
                param.requires_grad = False

        
    def forward(self,x):
        x = self.net(x)
        x_att, attention = self.att(x)
        x_pool = self.avg_pool(x_att)
        x_out = self.linear(torch.squeeze(x_pool))
        
               
        return x_out
    
class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention

class ImageModel(object):
    """
    ImageModel class
    """
    def __init__(self, batch_size, num_labels, model_name, conv_att = False, feature_extract=False):
        """ Initialization """
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.model_name = model_name
        self.cnn = self.model_name in {"resnet50","resnet152"}
        self.model_dir = MODEL_DIR_DICT[self.model_name]
        if not self.cnn:
            # feature extractor
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_dir)
        # model
        self.set_model(conv_att=conv_att, feature_extract=feature_extract)
        self.softmax = nn.Softmax(dim=1)
    
    def set_model(self, conv_att = False, feature_extract=False):
        if self.cnn:
            if conv_att:
                self.model = CNNAtt(model_dir=self.model_dir, model_name=self.model_name, num_labels=self.num_labels,
                feature_extract=feature_extract)   
            else:
                self.model = CNN(model_dir=self.model_dir, model_name=self.model_name, num_labels=self.num_labels,
                feature_extract=feature_extract) 
            
        else:
            if self.model_name == "vit":
                self.model = AutoModelForImageClassification.from_pretrained(self.model_dir, 
                        num_labels=self.num_labels)
            elif self.model_name == "beit":
                self.model = BEiT(self.model_dir, self.num_labels)
            else:
                self.model = DEiT(self.model_dir, self.num_labels)
        self.model.to(device)
        print(self.model)       
    
    def load_data(self, data, img_file_fmt, testing=False, task_name=None):
        train, y_vector_tr, val, y_vector_val, test, y_vector_te, class_weights, _ = prepare_data(data, self.num_labels, testing=testing)
        if self.cnn:
            tr_dataset = ImgOnlyCNN_Dataset(train.tweet_id.values,y_vector_tr,img_file_fmt,task_name=task_name)
            val_dataset = ImgOnlyCNN_Dataset(val.tweet_id.values,y_vector_val,img_file_fmt,task_name=task_name)
            te_dataset = ImgOnlyCNN_Dataset(test.tweet_id.values,y_vector_te,img_file_fmt,task_name=task_name)
        else:
            tr_dataset = ImgOnly_Dataset(train.tweet_id.values,y_vector_tr,self.feature_extractor,img_file_fmt,
                                         task_name=task_name)
            val_dataset = ImgOnly_Dataset(val.tweet_id.values,y_vector_val,self.feature_extractor,img_file_fmt,
                                          task_name=task_name)
            te_dataset = ImgOnly_Dataset(test.tweet_id.values,y_vector_te,self.feature_extractor,img_file_fmt,
                                         task_name=task_name)

        train_loader = DataLoader(tr_dataset, batch_size=self.batch_size)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(te_dataset, batch_size=self.batch_size)
        return train_loader, val_loader, test_loader, class_weights
    
    def train(self,dataloader,val_dataloader,epochs,loss_fn,lr,weight_decay,
              te_dataloader=None,model_path=None,val_filename=None,te_filename=None):
        #Initialize Optimizer
        named_parameters = self.model.named_parameters()
        optimizer_params = get_optimizer_params(named_parameters, weight_decay, lr)
        optimizer = optim.AdamW(optimizer_params, lr=lr)

        #optimizer= optim.Adam(self.model.parameters(), lr, weight_decay = weight_decay) 
        self.model.train()
        res_val, res_te = [], []
        for  epoch in range(epochs):
            print(epoch)
            loop=tqdm(enumerate(dataloader),leave=False,total=len(dataloader))
            for batch, dl in loop:
                pixel_values = torch.squeeze(dl['pixel_values'])
                if len(pixel_values.size())<4:
                    pixel_values = torch.unsqueeze(pixel_values,0)
                pixel_values = pixel_values.to(device)
                #print("pixel values",pixel_values.size())
                label=dl['labels'].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                if self.cnn:
                    output = self.model(pixel_values)
                    print("output", output.size())
                else:
                    output=self.model(
                            pixel_values=pixel_values,
                            )
                    if self.model_name == "vit":
                        output = output.logits

                # label
                label = label.type_as(output)
                # compute loss
                loss=loss_fn(output,label)
                # backward pass
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
        #return res_val

    def eval(self, dataloader, loss_fn):
        eval_acc = []
        eval_loss = []
        predictions = []
        labels = []
        data_ids = []
        loop=tqdm(enumerate(dataloader),leave=False,total=len(dataloader))
        self.model.eval()
        
        for batch, dl in loop:
            pixel_values = torch.squeeze(dl['pixel_values']).to(device)
            #print("pixel values",pixel_values.size())
            label = dl['labels'].to(device)
            data_id = dl['data_id'].to(device)
            # Compute logits
            with torch.no_grad():
                if self.cnn:
                    output = self.model(pixel_values)
                    #print("output", output.size())
                else:
                    output=self.model(
                        pixel_values=pixel_values,
                        )
                    if self.model_name == "vit":
                        output = output.logits
            # Compute loss
            label = label.type_as(output)
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

        print(f'test loss: {eval_loss:.4f} test acc: {(eval_acc):.4f}\n')
        
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

      
def main():
    #prepare_data_key(raw_data_path="textimage-data.csv")
    # data
    with open('failed_ids.txt') as f:
        lines = f.readlines()
    failed_ids = [int(x) for x in lines]
    data_key = pd.read_csv(PATH)
    data = data_key[["tweet_id",TASKS[task],"split"]].rename(columns={TASKS[task]:"label"})
    data = data[~data.tweet_id.isin(failed_ids)]
    # image model
    image_model = ImageModel(batch_size, num_labels, model_name)
    train_loader, test_loader, pos_weight = image_model.load_data(data, testing=testing, task_name=TASKS[task])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight = pos_weight)
    # Initialize Optimizer
    image_model.finetune(train_loader, epochs, loss_fn,lr,weight_decay)
    metric = BinaryF1Score(average="weighted")
    f1_score = image_model.eval(test_loader, loss_fn, metric)
    print("f1",f1_score) 

if __name__ == "__main__":
    main()  
   
    '''
    class ImgClassifier(pl.LightningModule):
        def __init__(self, model, lr: float = 2e-5, **kwargs): 
            super().__init__()
            self.save_hyperparameters('lr', *list(kwargs))
            self.model = model
            self.forward = self.model.forward 
            self.val_f1 = BinaryF1Score(average="weighted")

        def training_step(self, batch, batch_idx):
            outputs = self(**batch)
            self.log(f"train_loss", outputs.loss)
            return outputs.loss

        def validation_step(self, batch, batch_idx):
            outputs = self(**batch)
            self.log(f"val_loss", outputs.loss)
            acc = self.val_f1(outputs.logits.argmax(1), batch['labels'])
            self.log(f"val_f1", acc, prog_bar=True)
            return outputs.loss
        
        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), 
                                lr=self.hparams.lr,weight_decay = 0.00025)
                                
    THRESHOLD = 0.5
    # example
    img_file_fmt='./text-image/T{}.jpg'
    index = [2,3]
    image_id = data.tweet_id.values
    img = Image.open(img_file_fmt.format(image_id[index]))
    print("image", img_file_fmt.format(image_id[index]))
    inputs = feature_extractor(images=img, return_tensors="pt",data_format="channels_first")
    image = inputs.pixel_values
    print("size",image.size())
    print("ndim",image.ndim)
    import matplotlib.pyplot as plt 
    plt.imshow(img)
    sys.exit()
    
    
    label2id = {}
    id2label = {}
    for class_label in [0,1]:
        if task == 0:
            class_name = "text_is_represented" if class_label == 1 else "text_is_not_represented"
        else:
            # task == 1
            class_name = "image_adds" if class_label == 1 else "image_not_add"
        label2id[class_name] = str(class_label)
        id2label[str(class_label)] = class_name
    
    
    #pl.seed_everything(SEED)
    #classifier = ImgClassifier(model, lr=2e-5)
    #trainer = pl.Trainer(max_epochs=3)
    #trainer.fit(classifier, train_loader, test_loader)

    '''
