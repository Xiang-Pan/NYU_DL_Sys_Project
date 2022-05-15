'''
Author: Xiang Pan
Date: 2021-11-04 21:39:21
LastEditTime: 2021-11-04 21:52:00
LastEditors: Xiang Pan
Description: 
FilePath: /TextCLS/viz.py
xiangpan@nyu.edu
'''
import sys
import torch
from sklearn.manifold import TSNE 
import pandas as pd
import numpy as np
from transformers import AutoModel

sys.path.insert(0, '.')
sys.path.insert(0, '..')

import torch
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer


import os
from os import listdir
from os.path import isfile, join

def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            yield join(root, f)

def get_d():
    base = './cached_models/'
    d = {}
    for i in findAllFile(base):
        path = i
        method = path.split('/')[-3].split('_')[-2]
        if "deepset/roberta-base-squad2-covid" in path:
            model_name = "task-domain"
            backbone_name = "deepset/roberta-base-squad2-covid"
        elif "vinai/bertweet-covid19-base-cased" in path:
            model_name = "domain"
            backbone_name = "vinai/bertweet-covid19-base-cased"
        elif "deepset/roberta-base-squad2" in path:
            model_name = "task"
            backbone_name = "deepset/roberta-base-squad2"
        else:
            model_name = "baseline"
            backbone_name = "roberta-base"
        
        iteration = path.split("/")[-3][-1]
        log_name = model_name + "_" + method + "_" + iteration
        # print(log_name)
        d[log_name] = (path, backbone_name, method)
    print(d)
    return d


# args
from options import *

# task parts
from task_models.text_cls_model import TextCLSLightningModule
from task_datasets.covidqcls_dataset import CovidQCLSDataset


from argparse import Namespace

# log_name = "task_fold_random_0"
d = get_d()

for k, v in d.items():
    log_name = k
    ckpt_path, backbone_name, method = v
    args = {"backbone_name": backbone_name, "num_labels": 16, "tokenizer_name": "roberta-base", "batch_size":32, "load_classifier": None}
    args = Namespace(**args)
    model = TextCLSLightningModule.load_from_checkpoint(ckpt_path, args=args)

    eb_model = AutoModel.from_pretrained(backbone_name)
    eb_model.load_state_dict(model.net.roberta.state_dict(), strict=False)
    eb_model = eb_model.to(torch.device("cuda"))

    train_dataset = CovidQCLSDataset(tokenizer_name="roberta-base", split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)

    l = []
    label_list = []
    count = 0
    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch[0], batch[1], batch[2]
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        outputs = eb_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs["pooler_output"]
        l.append(embeddings.cpu().detach())
        label_list.append(labels.cpu().detach())
        count += 1

    x = torch.cat(l).numpy()
    y = torch.cat(label_list)

    np.save(f"./viz/feature_{log_name}",x)
    np.save(f"./viz/label_{log_name}",y)


    tsne = TSNE(n_components=2, init='random', perplexity=30, early_exaggeration=100) 
    X_tsne = tsne.fit_transform(x) 
    X_tsne_data = np.vstack((X_tsne.T, y)).T 
    df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'class']) 
    df_tsne.head()
    df_tsne['class'] = df_tsne['class'].astype(int)
    df_tsne.to_csv(f"./viz/tsne_{log_name}.csv", index=False)


    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(8, 8)) 
    sns.scatterplot(data=df_tsne, hue='class', x='Dim1', y='Dim2') 
    plt.savefig(f"./viz/tsne_{log_name}_fig")




