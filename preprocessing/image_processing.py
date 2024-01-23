import sys
sys.path.append("../")
from models.utils import prepare_data
from models.datasets import Lxmert_Dataset
from models.config import Config, PATH, TASKS, IMG_FMT, MODEL_DIR_DICT
import torch
from transformers import LxmertTokenizer
from lxmert_scripts.modeling_frcnn import GeneralizedRCNN
from lxmert_scripts.utils import Config as Lxmert_Config
from lxmert_scripts.processing_image import Preprocess

print("starting")
testing = True
task = 3
model_name = "lxmert"
model_dir = MODEL_DIR_DICT[model_name]
tokenizer = LxmertTokenizer.from_pretrained(model_dir)
frcnn_cfg = Lxmert_Config.from_pretrained(MODEL_DIR_DICT["frcnn"])
frcnn = GeneralizedRCNN.from_pretrained(MODEL_DIR_DICT["frcnn"], config=frcnn_cfg)
image_preprocess = Preprocess(frcnn_cfg)
data_path = "../data/"

cfg = Config(task, PATH, TASKS,IMG_FMT, model_name=model_name)
train, y_vector_tr, val, y_vector_val, test, y_vector_te, class_weights = prepare_data(
    cfg.data, cfg.num_labels, testing=testing)  
print("load dataset")
tr_dataset = Lxmert_Dataset(
    train.tweet_id.values,train.text.values,y_vector_tr,tokenizer, cfg.max_length, 
    image_preprocess, frcnn, frcnn_cfg, cfg.img_fmt)
for i in range(5):
    print(i,"/",5)
    inputs = tr_dataset[i]
    normalized_boxes = torch.squeeze(inputs['normalized_boxes'])
    features = torch.squeeze(inputs['features'])
    data_id = inputs['data_id'].item()
    torch.save(normalized_boxes, data_path+'mvsa_img_feats/boxes/val_nbox_{}'.format(data_id))
    torch.save(features, data_path+'mvsa_img_feats/features/val_feat_{}'.format(data_id))
print("done")