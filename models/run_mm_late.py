# TODO EVAL, MODels
import sys
import pandas as pd
import numpy as np
from mm_late import MMLate_Model
import argparse
from config import Config, results_dir_mm_late, IMAGE_ADDS
from utils import compute_metrics, agg_metrics_val
import torch.nn as nn
import torch
# ------ LOGGING-----------------------------------------------------
import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S',
                   level=logging.INFO,
                   )
logger = logging.getLogger(__name__)

# ------ ARGS -------------------------------------------------------
parser = argparse.ArgumentParser(description='run late fusion models')
parser.add_argument('--txt_model_name', type=str, choices=["bert","bernice","bertweet","roberta"],help='model name')
parser.add_argument('--img_model_name', type=str, choices=["vit","beit","deit","resnet50", "resnet152"],help='model name')
parser.add_argument('--fusion_name', type=str,choices=["xatt", "concat", "attention","concat_cnn", "aspect-att","gmu"], help='fusion method')
parser.add_argument('--use_clip_loss', action='store_true', help='use contrastive Loss')
parser.add_argument('--use_tim_loss', action='store_true', help='use TIM Loss')
parser.add_argument('--use_iadds_loss', action='store_true', help='use image-adds loss')
parser.add_argument('--beta_iadds', type=float, default=0.1, help='hyperparameter for iadds loss')
parser.add_argument('--beta_itc', type=float, default=0.1, help='hyperparameter for itc loss')
parser.add_argument('--beta_itm', type=float, default=0.1, help='hyperparameter for itm loss')
parser.add_argument('--use_loss_correction', action='store_true', help='use Loss correction (only for binary cases)')
parser.add_argument('--task', type=int, choices=[0,1,2,3,4,5,6], help='task to run')
parser.add_argument('--epochs', type=int, default=2, help='number of epochs')
parser.add_argument('--weight_decay', type=float, default=0.00025, help='weight decay param')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate param')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout param')
parser.add_argument('--seed', type=int, default=30, help='manual seed')
parser.add_argument('--nsamples', type=int, default=-1, help='number of training samples')
parser.add_argument('--testing', action='store_true', help='testing sample')
parser.add_argument('--eval_txt_test', action='store_true', help='eval txt test')
parser.add_argument('--save_model', action='store_true', help='save model')
parser.add_argument('--load_saved_model', action='store_true', help='load saved model')
parser.add_argument('--save_preds', action='store_true', help='eval test')
parser.add_argument('--use_saved_features', action='store_true', help='use preprocessed features')

args = parser.parse_args()

# SEED
torch.manual_seed(args.seed)
np.random.seed(args.seed)
# results directory
results_dir = results_dir_mm_late
image_adds_path = IMAGE_ADDS.format(args.task)
if args.testing:
    results_dir += "testing/"

logger.info("Model: {}-{}, Task: {}, Fusion: {}, Testing: {}, PP Features: {}, ITC Loss: {}, TIM Loss: {}, beta_itc: {}, beta_itm: {}, NSamples: {}, seed: {}".format(
        args.txt_model_name,args.img_model_name, args.task, args.fusion_name,
        args.testing,args.use_saved_features,
        args.use_clip_loss,args.use_tim_loss,
        args.beta_itc,args.beta_itm, 
        args.nsamples,
        args.seed))
# ------------------------------------------------------------------

def main():

    logger.info("Loading model and data")
    cfg = Config(args)
    
    # model
    mm_model = MMLate_Model(
        cfg, 
        args.txt_model_name, 
        args.img_model_name,
        args.fusion_name,
        multilabel=cfg.multilabel,
        )
    # data loaders
    train_loader, val_loader, test_loader, weight, txt_te_loader = mm_model.load_data(
        cfg.data, cfg.img_fmt, testing=args.testing, nsamples=args.nsamples, 
        saved_features=args.use_saved_features,
        task_name=cfg.task_name,
        eval_txt_test=args.eval_txt_test,
        )
    loss_fn = nn.CrossEntropyLoss(weight = weight) if not cfg.multilabel else nn.BCEWithLogitsLoss(pos_weight=weight)
    # file names
    model_path = None
    loss_str = cfg.loss_str
    nsamples_str = "" if args.nsamples == -1 else "N"+str(args.nsamples)+"_"
    if args.save_model or args.load_saved_model:   
        model_path = results_dir+"{}-{}-{}_task{}_seed{}_{}_{}net.pth".format(
            args.txt_model_name,args.img_model_name,args.fusion_name,args.task,args.seed,loss_str,nsamples_str) 
    val_filename = results_dir + "{}-{}-{}_task{}_seed{}_{}_{}metrics_val.csv".format(
        args.txt_model_name,args.img_model_name,args.fusion_name,args.task,args.seed,loss_str,nsamples_str)
    te_filename = results_dir + "{}-{}-{}_task{}_seed{}_{}_{}metrics_test.csv".format(
        args.txt_model_name,args.img_model_name,args.fusion_name,args.task,args.seed,loss_str,nsamples_str)
    tim_loss_fn = nn.CrossEntropyLoss() if cfg.use_tim_loss else None
    iadds_loss_fn = nn.CrossEntropyLoss() if cfg.use_iadds_loss else None
    if not args.load_saved_model:
        # train
        logger.info("Training")
        mm_model.train(
            train_loader,
            val_loader,
            args.epochs,
            loss_fn,
            cfg.lr,
            cfg.weight_decay,
            tim_loss_fn=tim_loss_fn,
            iadds_loss_fn=iadds_loss_fn,
            te_dataloader=test_loader,
            model_path=model_path,
            val_filename=val_filename, 
            te_filename=te_filename
            )
  
        if args.save_preds:
            predictions = mm_model.eval(test_loader,loss_fn,tim_loss_fn=tim_loss_fn,iadds_loss_fn=iadds_loss_fn)
            te_pred_df = pd.DataFrame(data={
                "data_id": predictions["data_id"].tolist(),
                "label": predictions["labels"].tolist(),
                "prediction":predictions["predictions"].tolist()
            })
            preds_filename = "{}-{}-{}_task{}_seed{}_{}_{}preds.csv".format(
                args.txt_model_name,args.img_model_name,args.fusion_name,
                args.task,args.seed,loss_str,nsamples_str)
            te_pred_df.to_csv(results_dir+preds_filename,index=False)
            logger.info("{} saved".format(results_dir+preds_filename))
        if args.eval_txt_test:
            # evaluate model (test) of last epochs model
            logger.info("Evaluate and compute metrics (txt test)")
            predictions = mm_model.eval(txt_te_loader,loss_fn,tim_loss_fn=tim_loss_fn,iadds_loss_fn=iadds_loss_fn)
            metrics = compute_metrics(predictions,cfg.num_labels)
            # save predictions and metrics
            # predictions
            pred_df = pd.DataFrame(data={
                "data_id": predictions["data_id"].tolist(),
                "label": predictions["labels"].tolist(),
                "prediction":predictions["predictions"].tolist()
            })
            preds_filename = "{}-{}-{}_task{}_seed{}_{}_{}preds_txt.csv".format(
                args.txt_model_name,args.img_model_name,args.fusion_name,
                args.task,args.seed,loss_str,nsamples_str)
            pred_df.to_csv(results_dir+preds_filename,index=False)
            logger.info("{} saved".format(preds_filename))
            # metrics
            metrics_pd = pd.DataFrame(metrics)
            res_filename = "{}-{}-{}_task{}_seed{}_{}_{}metrics_txt.csv".format(
                args.txt_model_name,args.img_model_name,args.fusion_name,
                args.task,args.seed,loss_str,nsamples_str)
            metrics_pd.to_csv(results_dir+res_filename,index=False)
            logger.info("{} saved".format(res_filename))
    
            #######################################################################
    else:
        # load pretrained model
        mm_model.load_saved_model(model_path)
        print("model loaded")
        # evaluate model 
        logger.info("Evaluate and compute metrics (test)")
        predictions = mm_model.eval(test_loader,loss_fn,tim_loss_fn=tim_loss_fn,iadds_loss_fn=iadds_loss_fn)
        
        # save predictions and metrics
        # predictions
        pred_df = pd.DataFrame(data={
            "data_id": predictions["data_id"].tolist(),
            "label": predictions["labels"].tolist(),
            "prediction":predictions["predictions"].tolist(),
            #"soft-0": predictions["soft_pred_0"].tolist(),
            #"soft-1": predictions["soft_pred_1"].tolist()
        })
        preds_filename = "{}-{}-{}_task{}_seed{}_{}_{}preds_lm.csv".format(
                args.txt_model_name,args.img_model_name,args.fusion_name,
                args.task,args.seed,loss_str,nsamples_str)
        pred_df.to_csv(results_dir+preds_filename,index=False)
        logger.info("{} saved".format(preds_filename))
        
        # metrics
        metrics = compute_metrics(predictions,cfg.num_labels, multilabel=cfg.multilabel)        
        metrics_pd = pd.DataFrame(metrics)
        res_filename = "{}-{}-{}_task{}_seed{}_{}_{}metrics_lm.csv".format(
            args.txt_model_name,args.img_model_name,args.fusion_name,
            args.task,args.seed,loss_str,nsamples_str)
        metrics_pd.to_csv(results_dir+res_filename,index=False)
        logger.info("{} saved".format(results_dir+res_filename))

    
    
   
    
    logger.info("Done!")


    
    

if __name__ == "__main__":
    main()  
 