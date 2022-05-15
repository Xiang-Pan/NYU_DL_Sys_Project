import argparse


def str2bool(str):
    return True if str.lower() == 'true' else False

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default="covidqcls_finetune", type=str, help="task name")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("--load_model", default=None, type=str, help="load model")
    parser.add_argument("--load_classifier", default=None, type=str, help="load model")
    parser.add_argument("--log_name", default=None, type=str, help="load model")



    # few_shots
    parser.add_argument("--num_shots", default=None, type=int, help="num_shots")
    return parser

