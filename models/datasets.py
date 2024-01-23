import sys
sys.path.append("../preprocessing/")
import torch.nn as nn
import torch
from text_processing import Tweet_Preprocessing
from PIL import Image
from torch.utils.data import Dataset
from utils import to_tensor_and_normalize, get_image_transforms
from config import DATA_PATH

class TxtOnly_Dataset(Dataset):
    def __init__(self, model_name, data_ids, text, labels, tokenizer, max_length, task_name, normalization = True):
        self.model_name = model_name
        self.data_ids = data_ids
        self.task_name = task_name
        if self.task_name == "poi":
            self.data_ids_num = [float(x.split("_")[0]) for x in self.data_ids]
        elif self.task_name in {"polid","poladv"}:
            self.data_ids_num = [float(x[2:]) for x in self.data_ids]
        else:
            self.data_ids_num = self.data_ids
        #self.data_ids_num = [float(x.split("_")[0]) for x in self.data_ids] if self.task_name == "poi" else self.data_ids
        self.labels = labels
        self.text = text
        self.tokenizer=tokenizer
        self.max_length=max_length
        self.normalization = normalization
        if self.normalization:
            self.tweet_preprocessing = Tweet_Preprocessing()            

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        
        text = self.tweet_preprocessing.normalizeTweet(self.text[index]) if self.normalization else self.text

        inputs = self.tokenizer.encode_plus(
            text ,
            None,
            padding='max_length',
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_length,
            #padding=True, 
            truncation=True
        )

        new_inputs = {}

        ids = inputs["input_ids"]
        new_inputs["ids"] = torch.tensor(ids, dtype=torch.long)
        if self.model_name not in  {"roberta","bernice"}:
            token_type_ids = inputs["token_type_ids"]
            new_inputs['token_type_ids'] = torch.tensor(token_type_ids, dtype=torch.long)
        mask = inputs["attention_mask"]
        new_inputs['mask'] = torch.tensor(mask, dtype=torch.long)
        new_inputs['target'] = torch.tensor(self.labels[index], dtype=torch.long)
        new_inputs['data_id'] = torch.tensor(self.data_ids_num[index], dtype=torch.long)

        return new_inputs


class ImgOnly_Dataset(Dataset):
    def __init__(self, data_ids, labels, feature_extractor, img_file_fmt, task_name):
        self.data_ids = data_ids
        self.task_name = task_name
        if self.task_name == "poi":
            self.data_ids_num = [float(x.split("_")[0]) for x in self.data_ids]
        elif self.task_name in {"polid","poladv"}:
            self.data_ids_num = [float(x[2:]) for x in self.data_ids]
        else:
            self.data_ids_num = self.data_ids
        #self.data_ids_num = [float(x.split("_")[0]) for x in self.data_ids] if self.task_name == "poi" else self.data_ids
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.img_file_fmt = img_file_fmt

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        try:
            img = Image.open(self.img_file_fmt.format(self.data_ids[index])).convert("RGB")
        except:
            img = Image.open(self.img_file_fmt.replace("jpg","png").format(self.data_ids[index])).convert("RGB")

        encodings = self.feature_extractor(img, return_tensors="pt", data_format="channels_first")
        encodings['labels'] = torch.tensor(self.labels[index], dtype=torch.long)
        encodings["data_id"] =  torch.tensor(self.data_ids_num[index], dtype=torch.long)
        return encodings  


class ImgOnlyCNN_Dataset(Dataset):
    def __init__(self, data_ids, labels,img_file_fmt, task_name):
        self.data_ids = data_ids
        self.task_name = task_name
        if self.task_name == "poi":
            self.data_ids_num = [float(x.split("_")[0]) for x in self.data_ids]
        elif self.task_name in {"polid","poladv"}:
            self.data_ids_num = [float(x[2:]) for x in self.data_ids]
        else:
            self.data_ids_num = self.data_ids
        self.labels = labels
        self.img_file_fmt = img_file_fmt


    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        encodings = {}
        try:
            img = Image.open(self.img_file_fmt.format(self.data_ids[index])).convert("RGB")
        except:
            img = Image.open(self.img_file_fmt.replace("jpg","png").format(self.data_ids[index])).convert("RGB")

        encodings["pixel_values"] = to_tensor_and_normalize(img)
        encodings['labels'] = torch.tensor(self.labels[index], dtype=torch.long)
        #print("task name", self.task_name)
        #print("ids",self.data_ids_num[index])
        encodings["data_id"] = torch.tensor(self.data_ids_num[index], dtype=torch.long)
        return encodings  

class MM_Dataset(Dataset):
    def __init__(self, data_ids, text, labels, processor, max_length, img_file_fmt=None, empty_image=None,
     normalization=True, saved_features=False, task_name=None, image_adds=None): 
        self.data_ids = data_ids
        self.task_name = task_name
        if self.task_name == "poi":
            self.data_ids_num = [float(x.split("_")[0]) for x in self.data_ids]
        elif self.task_name in {"polid","poladv"}:
            self.data_ids_num = [float(x[2:]) for x in self.data_ids]
        elif self.task_name == "fig":
            self.data_ids_num = [float(x.split(".")[0]) for x in self.data_ids]
        else:
            self.data_ids_num = self.data_ids
        #self.data_ids_num = [float(x.split("_")[0]) for x in self.data_ids] if self.task_name == "poi" else self.data_ids
        self.labels = labels
        self.text = text
        self.max_length=max_length
        self.normalization = normalization
        if self.normalization:
            self.tweet_preprocessing = Tweet_Preprocessing() 
        self.processor=processor
        self.img_file_fmt = img_file_fmt
        self.empty_image = empty_image
        self.saved_features = saved_features
        self.image_adds = image_adds
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        
        if self.saved_features:
            inputs_file_path = DATA_PATH+ '{}_img_feats/imgs/input_{}'.format(self.task_name,self.data_ids[index])
            #print("saved features" + inputs_file_path)
            inputs = torch.load(inputs_file_path)
        else:
            text = self.tweet_preprocessing.normalizeTweet(self.text[index]) if self.normalization else self.text[index]
            if self.empty_image == None:
                #print("not empy image")
                try:
                    image = Image.open(self.img_file_fmt.format(self.data_ids[index])).convert("RGB")
                except:
                    image = Image.open(self.img_file_fmt.replace("jpg","png").format(self.data_ids[index])).convert("RGB")
            else:
                #print("empty")
                image = Image.open(self.empty_image).convert("RGB")

            inputs = self.processor(
                text = text, 
                images = image, 
                padding= 'max_length',
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
                max_length=self.max_length
                )
                
        inputs['labels'] = torch.tensor(self.labels[index], dtype=torch.long)
        inputs['data_id'] = torch.tensor(self.data_ids_num[index], dtype=torch.long)

        if self.image_adds is not None:
            # Add prediction
            inputs["image_adds"] = torch.tensor(self.image_adds[index], dtype=torch.long)

        return inputs

class ViLT_Dataset(Dataset):
    def __init__(self, data_ids, text, labels, processor, max_length, img_file_fmt=None, empty_image=None,
     normalization=True, saved_features=False, task_name=None): 
        self.data_ids = data_ids
        self.task_name = task_name
        if self.task_name == "poi":
            self.data_ids_num = [float(x.split("_")[0]) for x in self.data_ids]
        elif self.task_name == "fig":
            self.data_ids_num = [float(x.split(".")[0]) for x in self.data_ids]
        elif self.task_name in {"polid","poladv"}:
            self.data_ids_num = [float(x[2:]) for x in self.data_ids]
        else:
            self.data_ids_num = self.data_ids

        self.labels = labels
        self.text = text
        self.max_length=max_length
        self.normalization = normalization
        if self.normalization:
            self.tweet_preprocessing = Tweet_Preprocessing() 
        self.processor=processor
        self.img_file_fmt = img_file_fmt
        self.empty_image = empty_image
        self.saved_features = saved_features
        
        
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        
        if self.saved_features:
            inputs_file_path = DATA_PATH+ '{}_img_feats/vilt/input_{}'.format(self.task_name,self.data_ids[index])
            #print("saved features" + inputs_file_path)
            inputs = torch.load(inputs_file_path)
        else:
            text = self.tweet_preprocessing.normalizeTweet(self.text[index]) if self.normalization else self.text[index]
            if self.empty_image == None:
                #print("not empy image")
                try:
                    image = Image.open(self.img_file_fmt.format(self.data_ids[index])).convert("RGB")
                except:
                    image = Image.open(self.img_file_fmt.replace("jpg","png").format(self.data_ids[index])).convert("RGB")
            else:
                #print("empty")
                image = Image.open(self.empty_image).convert("RGB")

            inputs = self.processor(
                text = text, 
                images = image, 
                padding= 'max_length',
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
                max_length=self.max_length
                )
                
        inputs['labels'] = torch.tensor(self.labels[index], dtype=torch.long)
        inputs['data_id'] = torch.tensor(self.data_ids_num[index], dtype=torch.long)

        return inputs

class Lxmert_Dataset(Dataset):
    def __init__(self, data_ids, text, labels, tokenizer, max_length, task_name, 
    normalization = True): 
        self.data_ids = data_ids
        self.task_name = task_name
        if self.task_name == "fig":
            self.data_ids_num = [float(x.split(".")[0]) for x in self.data_ids]
        else:
            self.data_ids_num = self.data_ids
        self.labels = labels
        self.text = text
        self.max_length=max_length
        self.normalization = normalization
        if self.normalization:
            self.tweet_preprocessing = Tweet_Preprocessing() 
        self.tokenizer=tokenizer
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        # text
        text = self.tweet_preprocessing.normalizeTweet(self.text[index]) if self.normalization else self.text
        inputs = self.tokenizer(
                text,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_token_type_ids=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
        # image (preprocessing is previously done to save time)
        data_id = self.data_ids[index]
        nbox_file_path = DATA_PATH + '{}_img_feats/boxes/nbox_{}'.format(self.task_name,data_id)
        feat_file_path = DATA_PATH + '{}_img_feats/features/feat_{}'.format(self.task_name,data_id)
        inputs['features'] =  torch.load(feat_file_path)
        inputs['normalized_boxes'] = torch.load(nbox_file_path)
        # labels & id
        inputs['labels'] = torch.tensor(self.labels[index], dtype=torch.long)
        inputs['data_id'] = torch.tensor(self.data_ids_num[index], dtype=torch.long)


        return inputs



class MM_CNN_Dataset(Dataset):
    def __init__(self, txt_model_name, data_ids, text, labels, tokenizer, max_length, 
                 img_file_fmt, normalization = True):
        self.txt_model_name = txt_model_name
        self.data_ids = data_ids
        self.labels = labels
        self.text = text
        self.max_length=max_length
        self.normalization = normalization
        if self.normalization:
            self.tweet_preprocessing = Tweet_Preprocessing() 
        self.tokenizer=tokenizer    
        self.img_file_fmt = img_file_fmt
        self.transforms = get_image_transforms()     
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        # text
        text = self.tweet_preprocessing.normalizeTweet(self.text[index]) if self.normalization else self.text
        inputs = self.tokenizer.encode_plus(
            text ,
            None,
            padding='max_length',
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_length,
            truncation=True
        )
        new_inputs = {}
        ids = inputs["input_ids"]
        new_inputs["ids"] = torch.tensor(ids, dtype=torch.long)
        if self.txt_model_name not in  {"roberta","bernice"}:
            token_type_ids = inputs["token_type_ids"]
            new_inputs['token_type_ids'] = torch.tensor(token_type_ids, dtype=torch.long)
        mask = inputs["attention_mask"]
        new_inputs['mask'] = torch.tensor(mask, dtype=torch.long)
        #image
        try:
            image = Image.open(self.img_file_fmt.format(self.data_ids[index])).convert("RGB")
        except:
            image = Image.open(self.img_file_fmt.replace("jpg","png").format(self.data_ids[index])).convert("RGB")

        new_inputs["pixel_values"] = self.transforms(image) #to_tensor_and_normalize(image)
        # labels
        new_inputs['labels'] = torch.tensor(self.labels[index], dtype=torch.long)
        new_inputs['data_id'] = torch.tensor(self.data_ids[index], dtype=torch.long)

        return new_inputs


class MMBT_Dataset(Dataset):
    def __init__(self, data_ids, text, labels, tokenizer, max_length, img_file_fmt, normalization=True, task_name=None): 
        
        self.data_ids = data_ids
        self.task_name = task_name
        if self.task_name == "fig":
            self.data_ids_num = [float(x.split(".")[0]) for x in self.data_ids]
        else:
            self.data_ids_num = self.data_ids
        self.labels = labels
        self.text = text
        self.max_seq_length = max_length
        self.normalization = normalization
        if self.normalization:
            self.tweet_preprocessing = Tweet_Preprocessing() 
        self.tokenizer = tokenizer
        self.img_file_fmt = img_file_fmt
        self.transforms = get_image_transforms()
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        text = self.tweet_preprocessing.normalizeTweet(self.text[index]) if self.normalization else self.text
        sentence = torch.LongTensor(self.tokenizer.encode(text, add_special_tokens=True))
        start_token, sentence, end_token = sentence[0], sentence[1:-1], sentence[-1]
        sentence = sentence[: self.max_seq_length]

        label = torch.tensor(self.labels[index], dtype=torch.long)
        try:
            image = Image.open(self.img_file_fmt.format(self.data_ids[index])).convert("RGB")
        except:
            image = Image.open(self.img_file_fmt.replace("jpg","png").format(self.data_ids[index])).convert("RGB")

        
        image = self.transforms(image)
        data_ids = torch.tensor(self.data_ids_num[index], dtype=torch.long)


        return {
            "image_start_token": start_token,
            "image_end_token": end_token,
            "sentence": sentence,
            "image": image,
            "label": label,
            "data_id": data_ids
        }
    
    
