from email.policy import default
import os
import sys
p = os.path.abspath('.')
sys.path.insert(0, p)
p = os.path.abspath('..')
sys.path.insert(1, p)

import torch
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from task_models.text_cls_model import TextCLSLightningModule
from task_datasets.covidqcls_dataset import CovidQCLSDataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import numpy as np

from scipy.stats import entropy
import random
import wandb

def lc_sample(all_preds, count):
    max_probs = [(p.max().item(), idx, p.argmax().item()) for idx, p in enumerate(all_preds)]
    max_probs.sort()
    return max_probs[:count+1], max_probs[count+1:]

def entropy_sample(all_preds, count):
    max_probs = [(entropy(p.detach().cpu().numpy()), idx, p.argmax().item()) for idx, p in enumerate(all_preds)]
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
    parser.add_argument("--al_fold", type=int, default=5)
    return parser

# get args
from options import get_parser
parser = get_parser()
parser = TextCLSLightningModule.add_model_specific_args(parser=parser)
parser = Trainer.add_argparse_args(parser)
parser = add_al_args(parser)
args = parser.parse_args()

# get dataset
dataset = CovidQCLSDataset(tokenizer_name="roberta-base", split='train')
train_loader = DataLoader(dataset, batch_size=args.batch_size)

val_dataset = CovidQCLSDataset(tokenizer_name="roberta-base", split='val')
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

test_dataset = CovidQCLSDataset(tokenizer_name="roberta-base", split='test')
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

remainder_loader = train_loader
# remainder_loader = 

method = args.method
active_learning=True


args.num_shots = 5
num_classes = 16

iteration = args.num_shots

data_size = len(dataset)
data_batch = 16

# Init our model
model = TextCLSLightningModule(args=args)

def split_probs_class(all_probs, method, num_classes=16):
    # all_preds = all_probs.argmax(dim=-1)

    # entropy_sample
    all_entropy = [(entropy(p.detach().cpu().numpy()), idx, p.argmax().item()) for idx, p in enumerate(all_probs)]

    # margin_sample
    

    # lc_sample
    all_lc = [(p.max().item(), idx, p.argmax().item()) for idx, p in enumerate(all_probs)]

    # all_probs_class = [[] for _ in range(num_classes)]
    class_list = [[] for _ in range(num_classes)]
    for i in range(num_classes):
        if method == "lc":
            all_lc = [(p.max().item(), idx, p[i]) for idx, p in enumerate(all_probs)]
            class_list[i] = sorted(all_lc, key = lambda x: x[2], reverse=True)
        elif method == "entropy":
            all_entropy = [(entropy(p.detach().cpu().numpy()), idx, p[i]) for idx, p in enumerate(all_probs)]
            class_list[i] = sorted(all_margin, key = lambda x: x[2], reverse=True)
        elif method == "margin":
            all_margin = [((p.topk(2).values[0] - p.topk(2).values[1]).item(), idx, p[i]) for idx, p in enumerate(all_probs)]
            class_list[i] = sorted(all_entropy, key = lambda x: x[2], reverse=True)
        else:
            raise ValueError("Unknown method")
    print("class_list", class_list)
            
    return all_probs_class


if active_learning:
    # Initialize a trainer
    trainer = Trainer(
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        progress_bar_refresh_rate=20,
    )
    sample_dataset = []
    for i in range(iteration):
        all_probs = []
        all_probs = trainer.predict(model, remainder_loader)
        all_probs = torch.cat(all_probs)
        all_probs = all_probs.softmax(dim=-1)
        all_probs_class = split_probs_class(all_probs, method=method)
        break
        if method == 'lc':
            sample, remainder = lc_sample(all_preds, data_batch)
        elif method == 'entropy':
            sample, remainder = entropy_sample(all_preds, data_batch)
        elif method == 'margin':
            sample, remainder = margin_sample(all_preds, data_batch)
        elif method == 'random':
            # sample_list = []
            # remainder_list = []
            # for i in range(num_classes):
            #     cur_indexes = all_probs_class[i]
            #     cur_indexes = cur_indexes[:data_batch]
            #     sample_list.append(cur_indexes)
            #     remainder_list.append(cur_indexes)
            sample, remainder = random_sample(all_probs_class[i], data_batch)

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

        # early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.01, patience=20, verbose=False, mode="min")
        
        wandb_logger = WandbLogger(name=log_name, project="NYU_DL_Sys_Project", group=f"active_learning_fsl")
        model = TextCLSLightningModule(args=args)
        trainer = Trainer(
            gpus=args.gpus,
            max_epochs=args.max_epochs,
            progress_bar_refresh_rate=20,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
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
        max_epochs=args.max_epochs,
        progress_bar_refresh_rate=20
    )
    model.train()
    trainer.fit(model, train_loader)
    model.eval()
    trainer.test(model, test_loader)
