'''
Author: Xiang Pan
Date: 2022-03-14 00:59:56
LastEditTime: 2022-04-15 21:22:03
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
            self.dataset = load_dataset("XiangPan/CovidQCLS", split="train[:90]")
        elif split == "val":
            self.dataset = load_dataset("XiangPan/CovidQCLS", split="train[90:]")
        elif split == "test":
            self.dataset = load_dataset("XiangPan/CovidQCLS", split="test")
        else:
            self.dataset = load_dataset("XiangPan/CovidQCLS", split="train+test")
        self.task_mapping = {
            "x": "Question",
            "y": "cate_id"
        }
        print(self.dataset)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> None:
        text = self.dataset[idx][self.task_mapping["x"]]
        x = self.tokenizer(text, padding="max_length", truncation=True, max_length=128)
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]
        y = self.dataset[idx][self.task_mapping["y"]]
        return input_ids, attention_mask, y


def main():
    dataset = CovidQCLSDataset(tokenizer_name="roberta-base")
    print(dataset[0])

if __name__ == "__main__":
    main()