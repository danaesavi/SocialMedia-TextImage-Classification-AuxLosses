class Config(object):
    """
    Config class
    """
    
    def __init__(self, args,model_name=None, multimodal=True, txt=False):
        # data
        import pandas as pd
        import numpy as np
        #self.column_names = ["tweet_id","text","label","split"] if (not multimodal or (not args.use_iadds_loss)) else ["tweet_id","text","label","split","image_adds"]
        self.multilabel = True if args.task in {10} else False
        self.column_names = ["tweet_id","text","label","split"]
        data_key = pd.read_csv(PATH[args.task])
        if args.task < 2:
            print("all", data_key.split.value_counts())
            with open("../data/failed_ids.txt", "r") as f:
                failed_ids = f.readlines()
            failed_ids = {int(x.strip()) for x in failed_ids}
            data_key = data_key[~data_key.tweet_id.isin(failed_ids)]
            print("removed failed ids", data_key.split.value_counts())
            self.data = data_key[["tweet_id","text",TASKS[args.task],"split"]].rename(
                columns={TASKS[args.task]:"label"})
            self.num_labels = 2
            self.batch_size = 8
        elif args.task == 2:
            print("all", data_key.split.value_counts())
            with open("../data/failed_ids.txt", "r") as f:
                failed_ids = f.readlines()
            failed_ids = {int(x.strip()) for x in failed_ids}
            data_key = data_key[~data_key.tweet_id.isin(failed_ids)]
            print("removed failed ids", data_key.split.value_counts())
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
            #column_names = ["id","text","label","split"] if not multimodal or not args.use_iadds_loss else ["id","text","label","split", "image_adds"]
            column_names = ["id","text","label","split"] 
            self.data = data_key[column_names].rename(
                columns={"id":"tweet_id"})
            self.num_labels = 2
            self.batch_size = 16
        elif args.task == 6:
            # task 6 
            failed_ids = {715376758774575105, 687440182027497472, 918237307169378305}
            data_key = data_key[~data_key.tweet_id.isin(failed_ids)]
            print("removed failed ids", data_key.split.value_counts())
            self.data = data_key[self.column_names]
            self.num_labels = 2
            self.batch_size = 16
        elif args.task in {7,8,9}:
            # task 7,8,9
            self.data = data_key[["tweet_id","text","label","split"]]
            self.num_labels = 2
            self.batch_size = 16
        elif args.task == 10:
            # task 10 FIG MEMES
            data_key["text"] = data_key["text"].astype(str)
            self.data = data_key[["tweet_id","text","label","split"]]
            self.num_labels = 6  # MULTICLASS
            self.batch_size = 16
            # TODO 
            # SIGMOID INSTEAD OF SOFTMAX AND ROUND
            # preds = torch.sigmoid(torch.tensor(preds.tolist()))
            # pred_labels = np.round(preds)
            # CHANGE LOSS TO BCE
            # CHANGE METRICS TO F1 MEAN PER LABEL
            # https://github.com/UKPLab/emnlp2022-figmemes/blob/main/run_classification.py LINE 390
        elif args.task == 13:
            # Image Adds (T1) --> MVSA (T3)
            self.data = data_key[["tweet_id","text","label","split"]]
            self.num_labels = 2
            self.batch_size = 16
        elif args.task == 14:
            # Image Adds (T1) --> MHP (T4)
            self.data = data_key[["tweet_id","text","label","split"]]
            self.num_labels = 2
            self.batch_size = 8
        elif args.task == 15:
            # Image Adds (T1) --> MIC (T5)
            self.data = data_key[["id","text","label","split"]].rename(
                columns={"id":"tweet_id"})
            self.num_labels = 2
            self.batch_size = 16
        elif args.task == 16:
            # Image Adds (T1) --> MSD (T6)
            self.data = data_key[["tweet_id","text","label","split"]]
            self.num_labels = 2
            self.batch_size = 16
        elif args.task == 17:
            # task 17 POI
            failed_ids = {"968182575209631745_2018-02-26-17-54-48_1"}
            data_key = data_key[~data_key.tweet_id.isin(failed_ids)]
            self.data = data_key[["tweet_id","text","label","split"]]
            self.num_labels = 8
            self.batch_size = 16
        elif args.task in {18,19}:
            # task 18, 19 POLADS
            self.data = data_key[["tweet_id","text","label","split"]]
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
            #self.use_iadds_loss = args.use_iadds_loss
            self.use_iadds_loss = False
            self.beta_itc = args.beta_itc if self.use_clip_loss else None
            self.beta_itm = args.beta_itm if self.use_tim_loss else None
            #self.beta_iadds = args.beta_iadds if self.use_iadds_loss else None
            self.beta_iadds = None
            # loss string
            self.loss_str = ""
            if args.use_clip_loss:
                # ITC loss
                self.loss_str += "itc{}".format(str(self.beta_itc)) 
            if args.use_tim_loss:
                # TIM loss
                self.loss_str += "itm{}".format(str(self.beta_itm)) 
            #if args.use_iadds_loss:
            #    self.loss_str += "iadds{}".format(str(self.beta_iadds))
        self.use_loss_correction = False
        # if multimodal or txt:
        #     if args.use_loss_correction:
        #         self.loss_str = "lc"
        #     self.use_loss_correction = args.use_loss_correction

# CONSTANTS

txt_feat_size = 768
fixed_feat_size = 768
img_feat_size = 768
img_feat_size_cnn = 2048

T = [
    [0.82148041, 0.17851959],
    [0.092827  , 0.907173]
    ]

TASKS = {
    0:"text_is_represented",
    1:"image_adds",
    2:"tir",
    3:"mvsa",
    4:"mhp",
    5:"mic",
    6:"msd",
    7:"backwards",
    8:"forwards",
    9:"random",
    10: "fig",
    13:"image_adds_mvsa",
    14:"image_adds_mhp",
    15:"image_adds_mic",
    16:"image_adds_msd",
    17: "poi",
    18: "polid",
    19: "poladv"
}
# stanage - fastdata
DATA_PATH = "/mnt/parscratch/users/ac1dsv/TImRel/data/"
# jade
#DATA_PATH = "../data/"



PATH = {
    0: DATA_PATH + "data_key_imgtxt_random.csv",
    1: DATA_PATH + "data_key_imgtxt_random.csv",
    2: DATA_PATH + "data_key_imgtxt_random.csv",
    3: DATA_PATH + "data_key_mvsa.csv",
    4: DATA_PATH + "data_key_mhp.csv",
    5: DATA_PATH + "data_key_mic.csv",
    6: DATA_PATH + "data_key_msd.csv",
    7: DATA_PATH + "data_key_backwards.csv",
    8: DATA_PATH + "data_key_forwards.csv",
    9: DATA_PATH + "data_key_random.csv",
    10: DATA_PATH + "data_key_fig.csv",
    13: DATA_PATH + "data_key_mvsa.csv",
    14: DATA_PATH + "data_key_mhp.csv",
    17: DATA_PATH + "data_key_poi.csv",
    18: DATA_PATH + "data_key_polid.csv",
    19: DATA_PATH + "data_key_poladv.csv"
    }

IMG_FMT ={
    0: DATA_PATH + 'text-image/T{}.jpg',
    1: DATA_PATH + 'text-image/T{}.jpg',
    2: DATA_PATH + 'text-image/T{}.jpg',
    3: DATA_PATH + 'MVSA-Single/data/{}.jpg',
    4: DATA_PATH +"MHP/Data/Images/{}.jpg",
    5: DATA_PATH + "MIC/spc_imgs_twitter/{}_1.jpg",
    6: DATA_PATH +'MSD/dataset_image/{}.jpg',
    7: DATA_PATH + 'yida-images/sunmm7/{}.jpg',
    8: DATA_PATH + 'yida-images/sunmm7/{}.jpg',
    9: DATA_PATH + 'yida-images/sunmm7/{}.jpg',
    10: DATA_PATH + 'FIG/dataset/images/{}',
    13: DATA_PATH + 'MVSA-Single/data/{}.jpg',
    14: DATA_PATH +"MHP/Data/Images/{}.jpg",
    17: DATA_PATH +"POI/poi_imgs_twitter/{}.jpg",
    18: DATA_PATH +"POL/pol_images/PolAdsImages/{}.png",
    19: DATA_PATH +"POL/pol_images/PolAdsImages/{}.png"
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
#RES_PATH = "../results/"
RES_PATH = "/mnt/parscratch/users/ac1dsv/TImRel/results/"
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





