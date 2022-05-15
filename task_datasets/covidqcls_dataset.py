'''
Author: Xiang Pan
Date: 2022-03-14 00:59:56
LastEditTime: 2022-05-14 19:58:49
LastEditors: Xiang Pan
Description: 
FilePath: /NYU_DL_Sys_Project/task_datasets/covidqcls_dataset.py
@email: xiangpan@nyu.edu
'''
from torch.utils.data import Dataset
import pandas as pd
from datasets import load_dataset
from yaml import load
from transformers import RobertaTokenizer
import itertools
import torch

class CovidQCLSDataset(Dataset):
    def __init__(self, tokenizer_name, split="full") -> None:
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
        if split == "train":
            self.dataset = load_dataset("XiangPan/CovidQCLS", split="train[:90%]")
        elif split == "val":
            self.dataset = load_dataset("XiangPan/CovidQCLS", split="train[90%:]")
        elif split == "test":
            self.dataset = load_dataset("XiangPan/CovidQCLS", split="test")
        else:
            self.dataset = load_dataset("XiangPan/CovidQCLS", split="train+test")
        self.task_mapping = {
            "x": "Question",
            "y": "cate_id"
        }
        self.class_lists = None
        

    def __len__(self) -> int:
        return len(self.dataset)

    def get_num_labels(self):
        return 16

    def __getitem__(self, idx: int):
        text = self.dataset[idx][self.task_mapping["x"]]
        x = self.tokenizer(text, padding="max_length",truncation=True, max_length=128, return_tensors="pt")
        input_ids = x["input_ids"].squeeze(0)
        attention_mask = x["attention_mask"].squeeze(0)
        y = self.dataset[idx][self.task_mapping["y"]]
        return input_ids, attention_mask, y

    def get_class_lists(self):
        # loop over dataset
        self.class_lists = {}
        for idx in range(len(self.dataset)):
            y = self.dataset[idx][self.task_mapping["y"]]
            if y not in self.class_lists.keys():
                self.class_lists[y] = [idx]
            else:
                self.class_lists[y].append(idx)
        
    
    def sample_few_shots_by_class(self, class_index, num_samples):
        if self.class_lists is None:
            self.get_class_lists()
        return self.class_lists[class_index][:num_samples]
    
    def sample_few_shots(self, num_samples):
        return [self.sample_few_shots_by_class(i, num_samples) for i in range(self.get_num_labels())]


def main():
    dataset = CovidQCLSDataset(tokenizer_name="roberta-base")
    sample_idxs = dataset.sample_few_shots(num_samples=5)
    sample_idxs = list(itertools.chain.from_iterable(sample_idxs))
    sub_dataset = torch.utils.data.Subset(dataset, sample_idxs)
    

if __name__ == "__main__":
    main()