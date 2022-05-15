'''
Created by: Xiang Pan
Date: 2022-04-14 12:55:13
LastEditors: Xiang Pan
LastEditTime: 2022-05-14 19:51:36
Email: xiangpan@nyu.edu
FilePath: /NYU_DL_Sys_Project/covid_cls_finetune.py
Description: 
'''
import sys
import os
sys.path.insert(0, '.')
sys.path.insert(0, '..')

import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import wandb

from task_models.text_cls_model import TextCLSLightningModule
from task_datasets.covidqcls_dataset import CovidQCLSDataset

def main():
    from options import get_parser
    parser = get_parser()
    parser = TextCLSLightningModule.add_model_specific_args(parser=parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    log_name = "fine_tune"
    dirpath = "./cached_models/" + args.task_name + "/" + str(log_name)
    
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    checkpoint_callback = ModelCheckpoint(
        monitor="val/f1",
        dirpath=dirpath,
        filename="{epoch:02d}-{val/f1:.2f}",
        save_top_k=1,
        mode="max",
    )

    wandb_logger = WandbLogger(name=args.task_name, project="NYU_DL_Sys_Project")

    trainer = Trainer(logger=wandb_logger, gpus=[0], max_epochs=args.max_epochs, callbacks=[checkpoint_callback])

    model = TextCLSLightningModule(args)
    train_dataset = CovidQCLSDataset(tokenizer_name=args.tokenizer_name, split='train')
    val_dataset = CovidQCLSDataset(tokenizer_name=args.tokenizer_name, split='val')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    trainer.test(model, val_dataloader)

if __name__ == '__main__':
    args = main()
