import os
import sys
p = os.path.abspath('.')
sys.path.insert(0, p)
p = os.path.abspath('..')
sys.path.insert(1, p)

import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from task_models.text_cls_model import TextCLSLightningModule
from task_datasets.covidqcls_dataset import CovidQCLSDataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from scipy.stats import entropy
import random
import argparse
import wandb

def lc_sample(all_preds, count):
    max_probs = [(p.max().item(), idx, p.argmax().item()) for idx, p in enumerate(all_preds)]
    max_probs.sort()
    return max_probs[:count+1], max_probs[count+1:]

def entropy_sample(all_preds, count):
    max_probs = [(entropy(p.detach().numpy()), idx, p.argmax().item()) for idx, p in enumerate(all_preds)]
    max_probs.sort(reverse=True)
    return max_probs[:count+1], max_probs[count+1:]

def margin_sample(all_preds, count):
    max_probs = [((p.topk(2).values[0] - p.topk(2).values[1]).item(), idx, p.argmax().item()) for idx, p in enumerate(all_preds)]
    max_probs.sort()
    return max_probs[:count+1], max_probs[count+1:]

def random_sample(all_preds, count):
    max_probs = [(p, idx, 0) for idx, p in enumerate(all_preds)]
    random.shuffle(max_probs)
    return max_probs[:count+1], max_probs[count+1:]


def add_al_args(parser):
    # selection from all_preds
    parser.add_argument("--method", type=str, default="random", choices=["random", "entropy", "margin", "lc"])
    return parser


from options import get_parser
parser = get_parser()
parser = TextCLSLightningModule.add_model_specific_args(parser=parser)
parser = Trainer.add_argparse_args(parser)
parser = add_al_args(parser)
args = parser.parse_args()



dataset = CovidQCLSDataset(tokenizer_name="roberta-base", split='train')
train_loader = DataLoader(dataset, batch_size=args.batch_size)

val_dataset = CovidQCLSDataset(tokenizer_name="roberta-base", split='val')
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

test_dataset = CovidQCLSDataset(tokenizer_name="roberta-base", split='test')
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

remainder_loader = train_loader

method = args.method
active_learning=True

iteration = 5

data_size = len(dataset)
data_batch = int(data_size/iteration)

# Init our model
model = TextCLSLightningModule(args=args)

if active_learning:
    # Initialize a trainer
    trainer = Trainer(
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        progress_bar_refresh_rate=20,
    )
    sample_dataset = []
    for i in range(iteration):
        all_preds = []
        all_preds = trainer.predict(model, remainder_loader)
        all_preds = [p.softmax(dim=-1).cuda() for p in all_preds]
        all_preds = torch.cat(all_preds)
        

        if method == 'lc':
            sample, remainder = lc_sample(all_preds, data_batch)
        elif method == 'entropy':
            sample, remainder = entropy_sample(all_preds, data_batch)
        elif method == 'margin':
            sample, remainder = margin_sample(all_preds, data_batch)
        else:
            sample, remainder = random_sample(all_preds, data_batch)

        # Create a new trainloader
        sample_dataset += [dataset[s[1]] for s in sample]
        remainder_dataset = [dataset[r[1]] for r in remainder]
        sample_loader = DataLoader(sample_dataset, batch_size=args.batch_size)
        remainder_loader = DataLoader(remainder_dataset, batch_size=args.batch_size)

        # Train the model ⚡

        # Init our model
        # wandb_logger = WandbLogger(
        log_name = "_".join([args.backbone_name, args.method, str(i)])

        args.task_name = "active_learning"
        
        dirpath = "./cached_models/" + args.task_name + "/" + str(log_name)
        checkpoint_callback = ModelCheckpoint(
            monitor="val/f1",
            dirpath=dirpath,
            filename="{epoch:02d}-{val/f1:.2f}",
            save_top_k=1,
            mode="max",
        )

        early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.01, patience=20, verbose=False, mode="min")
        
        wandb_logger = WandbLogger(name=log_name, project="NYU_DL_Sys_Project", group="active_learning")
        model = TextCLSLightningModule(args=args)
        trainer = Trainer(
            gpus=args.gpus,
            max_epochs=args.max_epochs,
            progress_bar_refresh_rate=20,
            logger=wandb_logger,
            callbacks=[checkpoint_callback, early_stop_callback],
        )
        model.train()
        trainer.fit(model, sample_loader, val_loader)
        model.eval()
        trainer.test(model, test_loader)
        wandb.finish()
else:
    # Init our model
    model = TextCLSLightningModule(args=args)
    # Initialize a trainer
    early_stop_callback = EarlyStopping(monitor="training/acc", stopping_threshold=0.99, verbose=False)
    trainer = Trainer(
        gpus=1,
        max_epochs=epochs,
        progress_bar_refresh_rate=20
    )
    model.train()
    trainer.fit(model, train_loader)
    model.eval()
    trainer.test(model, test_loader)
    
    
