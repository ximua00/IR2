import argparse
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = argparse.ArgumentParser()
config.add_argument('-root', '--root_dir', required=False, type=str, default="../data/", help='Data root direcroty')
config.add_argument('-exp_name', '--exp_name', required=True, type=str, help='Experiment Name')
config.add_argument('-dataset', '--data_set_name', required=False, type=str, default="Digital_Music_5.json", help='Dataset')
config.add_argument('-length', '--review_length', required=False, type=int, default=80,help='Review Length')
config.add_argument('-n_reviews', '--num_reviews_per_user', required=False, type=int, default=20, help='num_reviews_per_user')
config.add_argument('-batch_size', '--batch_size', required=False, type=int, default=64, help='Batch size')
config.add_argument('-nlr', '--narre_learning_rate', required=False, type=float, default=0.001, help='NARRE learning rate')
config.add_argument('-mlr', '--mrg_learning_rate', required=False, type=float, default=0.0003, help='MRG learning rate')
config.add_argument('-epochs', '--epochs', required=False, type=int, default=41, help='epochs')
config.add_argument('-wembed', '--word_embedding_size', required=False, type=int, default=200, help='Word embedding size')
config.add_argument('-iembed', '--id_embedding_size', required=False, type=int, default=200, help='Item/User embedding size')
config.add_argument('-efreq', '--eval_freq', required=False, type=int, default=1, help='Evaluation Frequency')
config.add_argument('-save_freq', '--save_freq', required=False, type=int, default=10, help='Model save Frequency')
config = config.parse_args()
#"Musical_Instruments_5.json"
#Digital_Music_5.json
