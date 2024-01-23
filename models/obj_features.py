import sys
sys.path.append("../")
from utils import prepare_data
from config import Config, PATH, TASKS, IMG_FMT, MODEL_DIR_DICT, EMPTY_IMG, DATA_PATH
import torch
from lxmert_scripts.modeling_frcnn import GeneralizedRCNN
from lxmert_scripts.utils import Config as Lxmert_Config
from lxmert_scripts.processing_image import Preprocess
from PIL import Image
import os
import pandas as pd
from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    AutoImageProcessor,
    ViTFeatureExtractor,
    AutoTokenizer,
    ViltProcessor
)
sys.path.append("../preprocessing/")
from text_processing import Tweet_Preprocessing
tweet_preprocessing = Tweet_Preprocessing() 
# ------ LOGGING-----------------------------------------------------
import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S',
                   level=logging.INFO,
                   )
logger = logging.getLogger(__name__)

# ARGUMENTS
testing = False
task = 10

logger.info("starting")
task_name = TASKS[task]

# prepare data
logger.info("load data")
# load data
data_key = pd.read_csv(PATH[task])
data_key["text"] = data_key["text"].astype(str)
if task == 17:
    failed_ids = {"968182575209631745_2018-02-26-17-54-48_1"}
    data_key = data_key[~data_key.tweet_id.isin(failed_ids)]
data_ids = data_key.tweet_id.values if task != 5 else data_key.id.values
texts = data_key.text.values
# preprocessing
if testing:
    N = 10
else:
    N = len(data_ids)

def preprocess_vilt(empty_image=None, normalization=True, max_length = 40):
    model_dir = MODEL_DIR_DICT["vilt"]
    processor = ViltProcessor.from_pretrained(model_dir)
    
    done_ids = os.listdir(DATA_PATH + "{}_img_feats/vilt".format(task_name))
    done_ids = {x.split("_")[1] for x in done_ids}
    logger.info("preprocessing {}/{} examples".format(N-len(done_ids),N))
    for index in range(N):
        data_id = data_ids[index]
        if data_id not in done_ids:
            # text
            #logger.info("data id: {}, text: {}".format(data_id, texts[index]))
            text = tweet_preprocessing.normalizeTweet(texts[index]) if normalization else texts[index]
            # img
            if empty_image == None:
                img_path = IMG_FMT[task].format(data_ids[index])
                if not os.path.exists(img_path):
                    img_path = img_path.replace("jpg","png")
                image = Image.open(img_path).convert("RGB")
            else:
                # empty image
                image = Image.open(EMPTY_IMG).convert("RGB")
            
            try:
                inputs = processor(
                    text = text, 
                    images = image, 
                    padding= 'max_length',
                    truncation=True,
                    add_special_tokens=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                    max_length=max_length,
                    return_token_type_ids=True
                    )  
                
                # save
                #print("item", inputs.keys())
                torch.save(inputs, DATA_PATH + '{}_img_feats/vilt/input_{}'.format(task_name, data_id))
                done_ids.add(data_id)
            except:
                logger.info("failed id {}".format(data_id))
        if index % 100 == 0:
            logger.info("{}/{}".format(index,N))
    logger.info("done")

def preprocess_mm(img_model_name,txt_model_name, empty_image=None, normalization=True, max_length = 128):
    img_model_dir = MODEL_DIR_DICT[img_model_name]
    txt_model_dir = MODEL_DIR_DICT[txt_model_name]
    tokenizer = AutoTokenizer.from_pretrained(txt_model_dir)
    feature_extractor = ViTFeatureExtractor.from_pretrained(img_model_dir)
    processor = VisionTextDualEncoderProcessor(feature_extractor, tokenizer)
    
    done_ids = os.listdir(DATA_PATH + "{}_img_feats/imgs".format(task_name))
    #done_ids = {int(x.split("_")[1]) for x in done_ids}
    done_ids = {x.split("_")[1] for x in done_ids}
    logger.info("preprocessing {}/{} examples".format(N-len(done_ids),N))
    for index in range(N):
        data_id = data_ids[index]
        if data_id not in done_ids:
            # text
            #print(index,texts[index])
            text = tweet_preprocessing.normalizeTweet(texts[index]) if normalization else texts[index]
            # img
            if empty_image == None:
                img_path = IMG_FMT[task].format(data_ids[index])
                if not os.path.exists(img_path):
                    img_path = img_path.replace("jpg","png")
                image = Image.open(img_path).convert("RGB")
            else:
                # empty image
                image = Image.open(EMPTY_IMG).convert("RGB")
            
            inputs = processor(
                text = text, 
                images = image, 
                padding= 'max_length',
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
                max_length=max_length
                )       
            # save
            torch.save(inputs, DATA_PATH +'{}_img_feats/imgs/input_{}'.format(task_name, data_id))
            done_ids.add(data_id)
        if index % 100 == 0:
            logger.info("{}/{}".format(index,N))
    logger.info("done")

def preprocess_obj():
    frcnn_cfg = Lxmert_Config.from_pretrained(MODEL_DIR_DICT["frcnn"])
    frcnn = GeneralizedRCNN.from_pretrained(MODEL_DIR_DICT["frcnn"], config=frcnn_cfg)
    image_preprocess = Preprocess(frcnn_cfg)
    done_ids = os.listdir(DATA_PATH + "{}_img_feats/boxes".format(task_name))
    done_ids = {x.split("_")[1] for x in done_ids}
    logger.info("preprocessing {}/{} examples".format(N-len(done_ids),N))
    for index in range(N):
        data_id = data_ids[index]
        if data_id not in done_ids:
            img_path = IMG_FMT[task].format(data_ids[index])
            if not os.path.exists(img_path):
                img_path = img_path.replace("jpg","png")

            images, sizes, scales_yx = image_preprocess(img_path)
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
            # save
            torch.save(normalized_boxes, DATA_PATH +'{}_img_feats/boxes/nbox_{}'.format(task_name, data_id))
            torch.save(features, DATA_PATH +'{}_img_feats/features/feat_{}'.format(task_name, data_id))
            done_ids.add(data_id)
        if index % 100 == 0:
            logger.info("{}/{}".format(index,N))
    logger.info("done")


def main():
    # mm late
    # img_model_name = "vit"
    # txt_model_name = "bernice"
    # preprocess_mm(img_model_name, txt_model_name, normalization=False)
    
    # lxmert
    preprocess_obj()

    # vit-bernice
    #img_model_name = "vit"
    #txt_model_name = "bernice"
    #preprocess_mm(img_model_name, txt_model_name, normalization=False)
    
    # vilt
    #preprocess_vilt()


if __name__ == "__main__":
    main()  




# DEPRECATED
# nbox_file_path = '../data/mvsa_img_feats/boxes/nbox_{}'.format(index)
# feat_file_path = '../data/mvsa_img_feats/features/feat_{}'.format(index)
# features =  torch.load(feat_file_path)
# normalized_boxes = torch.load(nbox_file_path)
# print(index)
# print("features", features.size())
# print("normalized_boxes", normalized_boxes.size())
