import os, sys

p = os.path.abspath('..')
sys.path.insert(1, p)

import os

import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy

from task_models.text_cls_model import TextCLSLightningModule
from task_datasets.covidqcls_dataset import CovidQCLSDataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from scipy.stats import entropy
import random
import argparse

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

parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="random")
args = parser.parse_args()


batch_size = 16
lr = 0.000005


dataset = CovidQCLSDataset(tokenizer_name="roberta-base", split='train')
train_loader = DataLoader(dataset, batch_size=batch_size)

val_dataset = CovidQCLSDataset(tokenizer_name="roberta-base", split='val')
val_loader = DataLoader(val_dataset, batch_size=batch_size)

test_dataset = CovidQCLSDataset(tokenizer_name="roberta-base", split='test')
test_loader = DataLoader(test_dataset, batch_size=batch_size)

remainder_loader = train_loader

method = args.method
active_learning=True

iteration = 5

epochs = 10
data_size = len(dataset)
data_batch = int(data_size/iteration)

# Init our model
model = TextCLSLightningModule(lr=lr)

if active_learning:
    
    # Initialize a trainer
    trainer = Trainer(
        gpus=1,
        max_epochs=epochs,
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
        sample_loader = DataLoader(sample_dataset, batch_size=batch_size)
        remainder_loader = DataLoader(remainder_dataset, batch_size=batch_size)

        # Train the model ⚡

        # Init our model
        model = TextCLSLightningModule(lr=lr)
        trainer = Trainer(
            gpus=1,
            max_epochs=epochs,
            progress_bar_refresh_rate=20
        )
        model.train()
        trainer.fit(model, sample_loader)
        model.eval()
        trainer.test(model, test_loader)
else:
    
    # Init our model
    model = TextCLSLightningModule(lr=lr)
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
    
    
