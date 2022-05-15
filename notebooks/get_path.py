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
        # print(i)
        path = i
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
        log_name = model_name + "_" + iteration
        # print(log_name)
        d[log_name] = (path, backbone_name)
    return d



if __name__ == '__main__':
    get_d()