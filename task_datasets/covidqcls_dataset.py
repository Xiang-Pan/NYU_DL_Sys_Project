'''
Author: Xiang Pan
Date: 2022-03-14 00:59:56
LastEditTime: 2022-05-14 19:04:44
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
        # delete_rows = []
        # for i, row in self.dataset.iterrows():
            
        
        # self.dataset = self.dataset[self.dataset["Question"] != None]

    def __len__(self) -> int:
        return len(self.dataset)

    def get_num_labels(self):
        return len(self.dataset.cate_id.unique())

    def __getitem__(self, idx: int):
        text = self.dataset[idx][self.task_mapping["x"]]
        x = self.tokenizer(text, padding="max_length",truncation=True, max_length=128, return_tensors="pt")
        input_ids = x["input_ids"].squeeze(0)
        attention_mask = x["attention_mask"].squeeze(0)
        y = self.dataset[idx][self.task_mapping["y"]]
        return input_ids, attention_mask, y


def main():
    dataset = CovidQCLSDataset(tokenizer_name="roberta-base")
    # print(dataset.get_num_labels())
    for data in dataset:
        print(data)
        assert data is not None
        # print(data)

if __name__ == "__main__":
    main()