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
    # args = {"backbone_name": backbone_name, "num_labels": 16, "tokenizer_name": "roberta-base", "batch_size":32, "load_classifier": None}
    # args = Namespace(**args)
    # model = TextCLSLightningModule.load_from_checkpoint(ckpt_path, args=args)

    # eb_model = AutoModel.from_pretrained(backbone_name)
    # eb_model.load_state_dict(model.net.roberta.state_dict(), strict=False)
    # eb_model = eb_model.to(torch.device("cuda"))

    # train_dataset = CovidQCLSDataset(tokenizer_name="roberta-base", split='train')
    # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)

    # l = []
    # label_list = []
    # count = 0
    # for batch in train_dataloader:
    #     input_ids, attention_mask, labels = batch[0], batch[1], batch[2]
    #     input_ids = input_ids.cuda()
    #     attention_mask = attention_mask.cuda()
    #     outputs = eb_model(input_ids=input_ids, attention_mask=attention_mask)
    #     embeddings = outputs["pooler_output"]
    #     l.append(embeddings.cpu().detach())
    #     label_list.append(labels.cpu().detach())
    #     count += 1

    # x = torch.cat(l).numpy()
    # y = torch.cat(label_list)

    # np.save(f"./viz/feature_{log_name}",x)
    # np.save(f"./viz/label_{log_name}",y)


    # tsne = TSNE(n_components=2, init='random', perplexity=30, early_exaggeration=100) 
    # X_tsne = tsne.fit_transform(x) 
    # X_tsne_data = np.vstack((X_tsne.T, y)).T 
    # df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'class']) 
    # df_tsne.head()
    # df_tsne['class'] = df_tsne['class'].astype(int)
    # df_tsne.to_csv(f"./viz/tsne_{log_name}.csv", index=False)


    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # plt.figure(figsize=(8, 8)) 
    # sns.scatterplot(data=df_tsne, hue='class', x='Dim1', y='Dim2') 
    # plt.savefig(f"./viz/tsne_{log_name}_fig")

    embeddings = np.load(f"./viz/feature_{log_name}.npy")
    y = np.load(f"./viz/label_{log_name}.npy")

    class_embeddings = [[] for i in range(16)]
    for i in range(len(embeddings)):
        class_embeddings[y[i]].append(embeddings[i])

    for i in range(16):
        class_embeddings[i] = np.array(class_embeddings[i])
        class_embeddings[i] = class_embeddings[i].mean(axis=0)

    # get distance matrix
    l2_dist_matrix = np.zeros((16, 16))
    cos_dist_matrix = np.zeros((16, 16))
    for i in range(16):
        for j in range(16):
            if class_embeddings[i] is None or class_embeddings[j] is None:
                l2_dist_matrix[i, j] = -1
                cos_dist_matrix[i, j] = -1
                continue
            l2_dist_matrix[i][j] = np.linalg.norm(class_embeddings[i] - class_embeddings[j])
            cos_dist_matrix[i][j] = np.dot(class_embeddings[i], class_embeddings[j]) / (np.linalg.norm(class_embeddings[i]) * np.linalg.norm(class_embeddings[j]))

    print(cos_dist_matrix, l2_dist_matrix)
    max_l2_dist = np.max(l2_dist_matrix)
    real_dist = l2_dist_matrix[l2_dist_matrix != 0]
    min_l2_dist = np.min(real_dist)
    mean_l2_dist = np.mean(l2_dist_matrix)
    print(max_l2_dist, min_l2_dist, mean_l2_dist)


    # do same for cosine distance
    max_cos_dist = np.max(cos_dist_matrix)
    real_dist = cos_dist_matrix[cos_dist_matrix != 0]
    min_cos_dist = np.min(real_dist)
    mean_cos_dist = np.mean(cos_dist_matrix)
    print(max_cos_dist, min_cos_dist, mean_cos_dist)


    # check df
    if not os.path.exists(f"./viz/dist_matrix.csv"):
        df = pd.DataFrame(columns=["log_name", "max_l2_dist", "min_l2_dist", "mean_l2_dist", "max_cos_dist", "min_cos_dist", "mean_cos_dist"])
        df.to_csv(f"./viz/dist_matrix.csv", index=False, mode="a")

    # append to df
    df = pd.read_csv("./viz/dist_matrix.csv")
    df.loc[len(df)] = [log_name, max_l2_dist, min_l2_dist, mean_l2_dist, max_cos_dist, min_cos_dist, mean_cos_dist]
    # two decimal places
    df = df.round(2)
    df.to_csv(f"./viz/dist_matrix.csv", index=False)





