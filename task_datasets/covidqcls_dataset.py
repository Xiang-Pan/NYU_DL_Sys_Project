'''
Author: Xiang Pan
Date: 2022-03-14 00:59:56
LastEditTime: 2022-03-14 01:01:31
LastEditors: Xiang Pan
Description: 
FilePath: /NYU_DL_Sys_Project/task_datasets/covidqcls_dataset.py
@email: xiangpan@nyu.edu
'''
from torch.utils.data import Dataset
import pandas as pd

class CovidQCLSDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> None:
        return None

def main():
    pd.read_csv("./cached_datasets/COVID-Q/data/train.csv")
    pass

if __name__ == "__main__":
    main()