class Config(object):
    """
    Config class
    """
    
    def __init__(self, args,model_name=None, multimodal=True, txt=False):
        # data
        import pandas as pd
        import numpy as np
        self.multilabel = True if args.task in {10} else False
        self.column_names = ["tweet_id","text","label","split"]
        data_key = pd.read_csv(PATH[args.task])
        if args.task < 2:
            self.data = data_key[["tweet_id","text",TASKS[args.task],"split"]].rename(
                columns={TASKS[args.task]:"label"})
            self.num_labels = 2
            self.batch_size = 8
        elif args.task == 2:
            data = data_key[["tweet_id","text","split"]]
            df_labels = data_key[["image_adds_text_repr","image_adds_text_notrepr",
            "image_notadds_text_repr","image_notadds_text_notrepr"]].to_numpy()
            label = np.argmax(df_labels, axis=1)
            data["label"] = label
            self.data = data[["tweet_id","text","label","split"]]
            self.num_labels = 4
            self.batch_size = 8
        elif args.task == 3:
            # task 3
            self.data = data_key[self.column_names]
            self.num_labels = 3
            self.batch_size = 16
        elif args.task == 4:
            # task 4
            self.data = data_key[self.column_names]
            self.num_labels = 4
            self.batch_size = 8
        elif args.task == 5:
            # task 5
            column_names = ["id","text","label","split"] 
            self.data = data_key[column_names].rename(
                columns={"id":"tweet_id"})
            self.num_labels = 2
            self.batch_size = 16
        elif args.task == 6:
            # task 6 
            self.data = data_key[self.column_names]
            self.num_labels = 2
            self.batch_size = 16
        self.img_fmt = IMG_FMT[args.task]
        self.task_name = TASKS[args.task]
        self.classes = CLASSES[args.task] if args.task in CLASSES else None
        self.dropout = args.dropout
        self.weight_decay = args.weight_decay
        self.lr = args.lr
        
        # Max length
        if model_name!= None and model_name == "vilt":
            self.max_length = 40
        else:
            self.max_length = 128
        if multimodal:
            # Aux Loss
            self.use_clip_loss = args.use_clip_loss
            self.use_tim_loss = args.use_tim_loss
            self.use_iadds_loss = False # deprecated
            self.beta_itc = args.beta_itc if self.use_clip_loss else None
            self.beta_itm = args.beta_itm if self.use_tim_loss else None
            self.beta_iadds = None # deprecated
            # loss string
            self.loss_str = ""
            if args.use_clip_loss:
                # ITC loss
                self.loss_str += "itc{}".format(str(self.beta_itc)) 
            if args.use_tim_loss:
                # TIM loss
                self.loss_str += "itm{}".format(str(self.beta_itm)) 
        self.use_loss_correction = False # deprecated
   

# CONSTANTS

txt_feat_size = 768
fixed_feat_size = 768
img_feat_size = 768
img_feat_size_cnn = 2048

TASKS = {
    0:"text_is_represented",
    1:"image_adds",
    2:"tir",
    3:"mvsa",
    4:"mhp",
    5:"mic",
    6:"msd",
}

DATA_PATH = "../data/"

PATH = {
    0: DATA_PATH + "data_key_imgtxt_random.csv",
    1: DATA_PATH + "data_key_imgtxt_random.csv",
    2: DATA_PATH + "data_key_imgtxt_random.csv",
    3: DATA_PATH + "data_key_mvsa.csv",
    4: DATA_PATH + "data_key_mhp.csv",
    5: DATA_PATH + "data_key_mic.csv",
    6: DATA_PATH + "data_key_msd.csv",
    }

IMG_FMT ={
    0: DATA_PATH + 'text-image/T{}.jpg',
    1: DATA_PATH + 'text-image/T{}.jpg',
    2: DATA_PATH + 'text-image/T{}.jpg',
    3: DATA_PATH + 'MVSA-Single/data/{}.jpg',
    4: DATA_PATH +"MHP/Data/Images/{}.jpg",
    5: DATA_PATH + "MIC/spc_imgs_twitter/{}_1.jpg",
    6: DATA_PATH +'MSD/dataset_image/{}.jpg',
}

CLASSES = {2:['image adds and text is represented',
              'image adds and text is not represented',
              'image does not add and text is represented',
              'image does not adds and text is not represented'],
            3:['neutral', 'positive', 'negative'],
            6:['not sarcastic','sarcastic']
            }

EMPTY_IMG = DATA_PATH + "MIC/empty_image.png"
TDATA5 = "../data/text_data_mic.csv"
metric_names = ["f1_weighted","f1_macro","precision_weighted","precision_macro","recall_weighted","recall_macro","loss"]
RES_PATH = "../results/"
results_dir_txt = RES_PATH + "txt_only/"
results_dir_img = RES_PATH + "img_only/"
results_dir_mm_early = RES_PATH + "mm_early/"
results_dir_mm_late = RES_PATH + "mm_late/"
results_dir_mmbt =  RES_PATH + "mmbt/"
IMAGE_ADDS = results_dir_mm_late + "bernice-vit-attention_task{}_seed30_preds_lm.csv"


MODEL_DIR_DICT = { 
    "bert": "../../../BERT-base/",            # "bert-base-uncased"
    "bertweet": "../../../BERTWEET-base/",    # "vinai/bertweet-base"
    "roberta": "../../../RoBERTa-base/",      # "roberta-base"
    "bernice": "../../../BERNICE/",           # "jhu-clsp/bernice"
    "vit": "../../../ViT/",                   # "google/vit-base-patch16-224-in21k"
    "beit": "../../../BEiT/",                 # "microsoft/beit-base-patch16-224-pt22k-ft22k"
    "deit": "../../../DEiT/",                 # deit-base-distilled-patch16-224
    "vilt": "../../../ViLT/",                 # "dandelin/vilt-b32-mlm"
    "lxmert": "unc-nlp/lxmert-base-uncased",
    "frcnn": "unc-nlp/frcnn-vg-finetuned",
    "resnet50": '../../../ConvModels/resnet50-0676ba61.pth',
    "resnet152": '../../../ConvModels/resnet152-394f9c45.pth'
    }





