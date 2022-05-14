import argparse


def str2bool(str):
    return True if str.lower() == 'true' else False

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default="covidqcls_finetune", type=str, help="task name")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    return parser

