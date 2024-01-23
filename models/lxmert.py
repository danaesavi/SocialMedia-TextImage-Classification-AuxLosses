import sys
sys.path.append("../")
import numpy as np
import torch
import torch.nn as nn
from lxmert_scripts.modeling_frcnn import GeneralizedRCNN
from lxmert_scripts.utils import Config
from lxmert_scripts.processing_image import Preprocess
from transformers import LxmertTokenizer, LxmertModel
from config import *




class Lxmert(nn.Module):
    def __init__(self, model_dir, max_length, dropout=0.1):
        super(Lxmert, self).__init__()
        self.model = LxmertModel.from_pretrained(model_dir)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(img_feat_size, num_labels) # 768 -> num_labels
        
    def forward(self,ids,mask,token_type_ids,features,normalized_boxes):
        output = self.model(
         input_ids=ids,
         attention_mask=mask,
         visual_feats=features,
         visual_pos=normalized_boxes,
         token_type_ids=token_type_ids,
         output_attentions=False,
         )
        pooled_output = output.pooled_output
        linear_output = self.linear(pooled_output)
        return linear_output
    
# image
frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)
image_preprocess = Preprocess(frcnn_cfg)
images, sizes, scales_yx = image_preprocess("../TImRel/data/text-image/T749760621688680448.jpg")
output_dict = frcnn(
     images,
     sizes,
     scales_yx=scales_yx,
     padding="max_detections",
     max_detections=frcnn_cfg.max_detections,
     return_tensors="pt",
 )
normalized_boxes = output_dict.get("normalized_boxes")
features = output_dict.get("roi_features")
# text
lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
text = "@USER Just chillin with Lando at my feet URL"
inputs = lxmert_tokenizer(
         [text],
         padding="max_length",
         max_length=20,
         truncation=True,
         return_token_type_ids=True,
         return_attention_mask=True,
         add_special_tokens=True,
         return_tensors="pt",
     )

# model
model = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
output = model(
         input_ids=inputs.input_ids,
         attention_mask=inputs.attention_mask,
         visual_feats=features,
         visual_pos=normalized_boxes,
         token_type_ids=inputs.token_type_ids,
         output_attentions=False,
     )

