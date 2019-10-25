import argparse
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = argparse.ArgumentParser()
config.add_argument('-root', '--root_dir', required=False, type=str, default="../data/", help='Data root direcroty')
config.add_argument('-dataset', '--data_set_name', required=False, type=str, default="Digital_Music_5.json", help='Dataset')
config.add_argument('-length', '--review_length', required=False, type=int, default=80,help='Review Length')
config.add_argument('-batch_size', '--batch_size', required=False, type=int, default=5, help='Batch size')
config.add_argument('-nlr', '--narre_learning_rate', required=False, type=float, default=0.001, help='NARRE learning rate')
config.add_argument('-mlr', '--mrg_learning_rate', required=False, type=float, default=0.0003, help='MRG learning rate')
config.add_argument('-epochs', '--epochs', required=False, type=int, default=2501, help='epochs')
config.add_argument('-wembed', '--word_embedding_size', required=False, type=int, default=100, help='Word embedding size')
config.add_argument('-iembed', '--id_embedding_size', required=False, type=int, default=100, help='Item/User embedding size')
config.add_argument('-efreq', '--eval_freq', required=False, type=int, default=50,help='Evaluation Frequency')
config = config.parse_args()