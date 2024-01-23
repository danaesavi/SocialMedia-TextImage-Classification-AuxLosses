import sys
import pandas as pd
import numpy as np
from text_only import TextModel
import argparse
from config import Config, results_dir_txt
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
parser = argparse.ArgumentParser(description='run text-only models')
parser.add_argument('--model_name', type=str, choices=["bert","bernice","bertweet","roberta"],help='model name')
parser.add_argument('--task', type=int, choices=[0,1,2,3,4,5,6,17,18,19], help='task to run')
parser.add_argument('--use_loss_correction', action='store_true', help='use Loss correction (only for binary cases)')
parser.add_argument('--epochs', type=int, default=2, help='number of epochs')
parser.add_argument('--weight_decay', type=float, default=0.00025, help='weight decay param')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate param')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout param')
parser.add_argument('--seed', type=int, default=30, help='manual seed')
parser.add_argument('--testing', action='store_true', help='testing sample')
parser.add_argument('--eval_txt_test', action='store_true', help='eval txt test')
parser.add_argument('--save_model', action='store_true', help='save model')
parser.add_argument('--save_preds', action='store_true', help='eval test')

args = parser.parse_args()
# SEED
torch.manual_seed(args.seed)
np.random.seed(args.seed)
# results directory
results_dir = results_dir_txt
if args.testing:
    results_dir += "testing/"

logger.info("Model: {}, Task: {}, Epochs: {}, LC:{}, seed: {}".format(
    args.model_name, args.task, args.epochs, args.use_loss_correction, args.seed))
# ------------------------------------------------------------------

def main():
    #prepare_data_key(raw_data_path="textimage-data.csv")
    logger.info("Loading model and data")
    cfg = Config(args, multimodal=False, txt=True)
    # model
    text_model = TextModel(cfg, args.model_name)
    train_loader, val_loader, test_loader, weight, txt_te_loader = text_model.load_data(
        cfg.data, testing=args.testing, eval_txt_test=args.eval_txt_test, task_name=cfg.task_name)
    loss_fn = nn.CrossEntropyLoss(weight = weight) if not args.use_loss_correction else nn.CrossEntropyLoss(weight = weight, reduction='none')
    model_path = None
    if args.save_model:
        model_path = results_dir+"{}_task{}_seed{}_net.pth".format(
            args.model_name,args.task,args.seed) 
    logger.info("Training")        
    val_filename = results_dir + "{}_task{}_seed{}_metrics_val.csv".format(
        args.model_name,args.task,args.seed)
    te_filename =  results_dir + "{}_task{}_seed{}_metrics_test.csv".format(
        args.model_name,args.task,args.seed)  
    text_model.train(train_loader,val_loader,args.epochs,loss_fn,cfg.lr,cfg.weight_decay,
    te_dataloader=test_loader,model_path=model_path,val_filename=val_filename, 
    te_filename=te_filename)

    if args.save_preds:
        predictions = text_model.eval(test_loader,loss_fn)
        te_pred_df = pd.DataFrame(data={
            "data_id": predictions["data_id"].tolist(),
            "label": predictions["labels"].tolist(),
            "prediction":predictions["predictions"].tolist()
        })
        preds_filename = "{}_task{}_seed{}_preds.csv".format(
                args.model_name,args.task,args.seed)
        te_pred_df.to_csv(results_dir+preds_filename,index=False)
        logger.info("{} saved".format(preds_filename))
    
    if args.eval_txt_test:
        # evaluate model (test) of last epochs model
        logger.info("Evaluate and compute metrics (txt test)")
        predictions = text_model.eval(txt_te_loader,loss_fn)
        metrics = compute_metrics(predictions,cfg.num_labels)
        # save predictions and metrics
        # predictions
        pred_df = pd.DataFrame(data={
            "data_id": predictions["data_id"].tolist(),
            "label": predictions["labels"].tolist(),
            "prediction":predictions["predictions"].tolist()
        })
        preds_filename = "{}_task{}_seed{}_preds_txt.csv".format(
                args.model_name,args.task,args.seed)
        pred_df.to_csv(results_dir+preds_filename,index=False)
        logger.info("{} saved".format(results_dir+preds_filename))
        # metrics
        metrics_pd = pd.DataFrame(metrics)
        res_filename = "{}_task{}_seed{}_metrics_txt.csv".format(
                args.model_name,args.task,args.seed)
        metrics_pd.to_csv(results_dir+res_filename,index=False)
        logger.info("{} saved".format(results_dir+res_filename))
  
    logger.info("Done!")
    

if __name__ == "__main__":
    main()  
 
