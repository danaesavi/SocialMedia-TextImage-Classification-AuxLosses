import sys
import pandas as pd
import numpy as np
from image_only import ImageModel
import argparse
from config import Config, results_dir_img
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
parser.add_argument('--model_name', type=str, choices=["vit","beit","deit","resnet50", "resnet152"],help='model name')
parser.add_argument('--conv_att', action='store_true', help='CNN ATT')
parser.add_argument('--feature_extract', action='store_true', help='feature_extract')
parser.add_argument('--task', type=int, choices=[0,1,2,3,4,5,6], help='task to run')
parser.add_argument('--epochs', type=int, default=2, help='number of epochs')
parser.add_argument('--weight_decay', type=float, default=0.00025, help='weight decay param')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate param')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout param')
parser.add_argument('--seed', type=int, default=30, help='manual seed')
parser.add_argument('--testing', action='store_true', help='testing sample')
parser.add_argument('--save_model', action='store_true', help='eval test')
parser.add_argument('--save_preds', action='store_true', help='eval test')

args = parser.parse_args()
# SEED
torch.manual_seed(args.seed)
np.random.seed(args.seed)
# results directory
results_dir = results_dir_img
if args.testing:
    results_dir += "testing/"

logger.info("Model: {}, Task: {}, feature extract: {}, conv att: {}, Epochs: {}, seed: {}".format(
        args.model_name, args.task, args.feature_extract, args.conv_att, args.epochs, 
        args.seed))
    
# ------------------------------------------------------------------

def main():
    logger.info("Loading model and data")
    cfg = Config(args, multimodal=False)
    # model
    image_model = ImageModel(cfg.batch_size, cfg.num_labels, args.model_name, 
    conv_att = args.conv_att, feature_extract=args.feature_extract)
    train_loader, val_loader, test_loader, weight = image_model.load_data(
        cfg.data, cfg.img_fmt, testing=args.testing, task_name=cfg.task_name)
    loss_fn = nn.CrossEntropyLoss(weight = weight)
    
    model_path = None
    if args.save_model:
        model_path = results_dir+"{}_task{}_seed{}_net.pth".format(
            args.model_name,args.task,args.seed) 
    
    logger.info("Training")
    val_filename = results_dir + "{}_task{}_seed{}_metrics_val.csv".format(
        args.model_name,args.task,args.seed)
    te_filename =  results_dir + "{}_task{}_seed{}_metrics_test.csv".format(
        args.model_name,args.task,args.seed)      
    image_model.train(train_loader,val_loader,args.epochs,loss_fn,cfg.lr,cfg.weight_decay,
                                te_dataloader=test_loader,model_path=model_path,
                             val_filename=val_filename, te_filename=te_filename)
   
    logger.info("{} saved!".format(val_filename))

    if args.save_preds:
        logger.info("Evaluate and compute metrics")
        # evaluate model
        predictions = image_model.eval(test_loader,loss_fn)
        # save predictions 
        pred_df = pd.DataFrame(data={
            "data_id": predictions["data_id"].tolist(),
            "label": predictions["labels"].tolist(),
            "prediction":predictions["predictions"].tolist()
        })
        preds_filename = results_dir+"{}_task{}_seed{}_preds.csv".format(
            args.model_name,args.task,args.seed)
        pred_df.to_csv(preds_filename,index=False)
        logger.info("{} saved".format(preds_filename))
    logger.info("Done!")
    

if __name__ == "__main__":
    main()  
 
