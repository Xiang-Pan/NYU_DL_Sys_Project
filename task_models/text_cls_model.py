'''
Created by: Xiang Pan
Date: 2022-04-15 21:21:21
LastEditors: Xiang Pan
LastEditTime: 2022-05-14 19:03:53
Email: xiangpan@nyu.edu
FilePath: /NYU_DL_Sys_Project/task_models/text_cls_model.py
Description: 
'''
from ast import parse
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from transformers import RobertaForSequenceClassification, AdamW, RobertaTokenizer, get_linear_schedule_with_warmup, RobertaModel, XLNetForSequenceClassification, AutoModelForSequenceClassification, AutoTokenizer
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
import copy


class TextCLSLightningModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        # assert args.lr
        self.args = args
        # if "/" not in args.backbone_name:
        self.net = AutoModelForSequenceClassification.from_pretrained(args.backbone_name, num_labels=args.num_labels)
        if self.args.load_classifier:
            self.net.classifier.load_state_dict(torch.load(self.args.load_classifier))

        self.train_acc_metric = torchmetrics.Accuracy()
        self.val_acc_metric = torchmetrics.Accuracy()
        self.test_acc_metric = torchmetrics.Accuracy()

        self.train_f1_metric = torchmetrics.F1Score()
        self.val_f1_metric = torchmetrics.F1Score()
        self.test_f1_metric = torchmetrics.F1Score()

        self.loss_func = nn.CrossEntropyLoss()
        self.is_raw_text_input = False
        if self.is_raw_text_input:
            self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
            
    def get_logits_and_loss(self, input_ids, attention_mask, labels):
        outputs = self.net(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs['logits']
        loss = outputs['loss']
        return logits, loss
    
    def training_step(self, batch, batch_idx, prefix='training'):
        if self.is_raw_text_input:
            texts = batch["text"]
            labels = batch["label"]
            texts = ['None' if v is None else v for v in texts]
            encoding = self.tokenizer(texts, return_tensors="pt", padding='max_length', truncation=True, max_length=128)
            input_ids, attention_mask = encoding['input_ids'].to(self.device), encoding['attention_mask'].to(self.device)
        else:
            input_ids, attention_mask, labels = batch

        logits, loss = self.get_logits_and_loss(input_ids, attention_mask, labels)
        acc = self.train_acc_metric(logits.softmax(dim=-1).cuda(), labels)
        f1 = self.train_f1_metric(logits.softmax(dim=-1).cuda(), labels)
        
        self.log(f'{prefix}/loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(f'{prefix}/acc', acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(f'{prefix}/f1', f1, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx, prefix='val'):
        if self.is_raw_text_input:
            texts = batch["text"]
            labels = batch["label"]
            texts = ['None' if v is None else v for v in texts]
            encoding = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            input_ids, attention_mask = encoding['input_ids'].to(self.device), encoding['attention_mask'].to(self.device)
        else:
            input_ids, attention_mask, labels = batch
        logits, loss = self.get_logits_and_loss(input_ids, attention_mask, labels)
        acc = self.val_acc_metric(logits.softmax(dim=-1).cuda(), labels)
        f1 = self.val_f1_metric(logits.softmax(dim=-1).cuda(), labels)
        self.log(f'{prefix}/loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(f'{prefix}/acc', acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(f'{prefix}/f1', f1, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def test_step(self, batch, batch_idx, prefix='test'):
        if self.is_raw_text_input:
            texts = batch["text"]
            labels = batch["label"]
            texts = ['None' if v is None else v for v in texts]
            encoding = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            input_ids, attention_mask = encoding['input_ids'].to(self.device), encoding['attention_mask'].to(self.device)
        else:
            input_ids, attention_mask, labels = batch
        logits, loss = self.get_logits_and_loss(input_ids, attention_mask, labels)
        acc = self.test_acc_metric(logits.softmax(dim=-1).cuda(), labels)
        f1 = self.test_f1_metric(logits.softmax(dim=-1).cuda(), labels)
        self.log(f'{prefix}/loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(f'{prefix}/acc', acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(f'{prefix}/f1', f1, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.net.parameters(), lr=self.args.lr, correct_bias=False)
    
    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--num_labels', type=int, default=16)
        parser.add_argument('--backbone_name', type=str, default='roberta-base')
        parser.add_argument('--tokenizer_name', type=str, default='roberta-base')
        return parser