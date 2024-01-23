import sys
import pandas as pd
import numpy as np
from mm_early import MMEarly_Model
import argparse
from config import Config, results_dir_mm_early
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
parser.add_argument('--model', type=str, choices=["vilt","lxmert"], help='model name')
parser.add_argument('--use_clip_loss', action='store_true', help='use CLIP Loss')
parser.add_argument('--beta_itc', type=float, default=0.1, help='hyperparameter for itc loss')
parser.add_argument('--beta_itm', type=float, default=0.1, help='hyperparameter for itm loss')
parser.add_argument('--use_tim_loss', action='store_true', help='use TIM Loss')
parser.add_argument('--use_loss_correction', action='store_true', help='use Loss correction (only for binary cases)')
parser.add_argument('--task', type=int, choices=[0,1,2,3,4,5,6,7,8,9,10,17,18,19], help='task to run')
parser.add_argument('--epochs', type=int, help='number of epochs')
parser.add_argument('--weight_decay', type=float, default=0.00025, help='weight decay param')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate param')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout param')
parser.add_argument('--seed', type=int, default=30, help='manual seed')
parser.add_argument('--testing', action='store_true', help='testing sample')
parser.add_argument('--evaltest', action='store_true', help='eval test')
parser.add_argument('--save_model', action='store_true', help='eval test')
parser.add_argument('--use_saved_features', action='store_true', help='use preprocessed features')
args = parser.parse_args()

# SEED
torch.manual_seed(args.seed)
np.random.seed(args.seed)
# results directory
results_dir = results_dir_mm_early
if args.testing:
    results_dir += "testing/"
model_name = args.model
logger.info("Model: {}, Task: {}, Epochs: {}, ITC loss: {}, TIM loss: {}, beta_itc: {}, beta_itm: {}, seed: {}".format(
    model_name, args.task, args.epochs, args.use_clip_loss,
    args.use_tim_loss, args.beta_itc, args.beta_itm, args.seed))
# ------------------------------------------------------------------

def main():
    logger.info("Loading model and data...")

    # Config class
    cfg = Config(args, model_name=model_name)

    # model
    mm_model = MMEarly_Model(cfg,model_name,multilabel=cfg.multilabel)

    # load data    
    train_loader, val_loader, test_loader, weight = mm_model.load_data(
        cfg.data,
        img_file_fmt=cfg.img_fmt,
        task_name=cfg.task_name,
        testing=args.testing,
        saved_features=args.use_saved_features,
        )
    loss_fn = nn.CrossEntropyLoss(weight = weight) if not cfg.multilabel else nn.BCEWithLogitsLoss(pos_weight=weight)


    # Model path to save model, test and dev filenames
    model_path = None
    loss_str = cfg.loss_str 
    if args.save_model:
        model_path = results_dir + "{}_task{}_seed{}_{}_net.pth".format(
            model_name,args.task,args.seed,loss_str) 
    val_filename = results_dir + "{}_task{}_seed{}_{}_metrics_val.csv".format(
        model_name,args.task,args.seed,loss_str)
    te_filename = results_dir + "{}_task{}_seed{}_{}_metrics_test.csv".format(
        model_name,args.task,args.seed,loss_str)
    
    # Train
    logger.info("Training...")
    tim_loss_fn = nn.CrossEntropyLoss() if cfg.use_tim_loss else None
    mm_model.train(train_loader,val_loader,args.epochs,loss_fn,cfg.lr,cfg.weight_decay,
    tim_loss_fn=tim_loss_fn,
    te_dataloader=test_loader,model_path=model_path,
    val_filename=val_filename, te_filename=te_filename)
 
    if args.evaltest:
        logger.info("Evaluate and compute metrics")
        # evaluate model (test)
        predictions = mm_model.eval(test_loader, loss_fn, tim_loss_fn=tim_loss_fn)
        metrics = compute_metrics(predictions,cfg.num_labels,multilabel=cfg.multilabel)
        print(metrics)
        print()
        # save predictions and metrics
        if not args.testing:
            # predictions
            pred_df = pd.DataFrame(data={
                "data_id": predictions["data_id"].tolist(),
                "label": predictions["labels"].tolist(),
                "prediction":predictions["predictions"].tolist()
            })
            preds_filename = "{}_task{}_seed{}_{}_preds.csv".format(
                model_name,args.task,args.seed,loss_str)
            pred_df.to_csv(results_dir+preds_filename,index=False)
            logger.info("{} saved".format(preds_filename))
            # metrics
            metrics_pd = pd.DataFrame(metrics)
            res_filename = "{}_task{}_seed{}_{}_metrics.csv".format(
                model_name,args.task,args.seed,loss_str)
            metrics_pd.to_csv(results_dir+res_filename,index=False)
            logger.info("{} saved".format(res_filename))
    logger.info("Done!")
    

if __name__ == "__main__":
    main()  
 