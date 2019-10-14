import torch
import argparse
from KGRAMSData import KGRAMSData
from model import KGRAMS
from torch.utils.data import DataLoader
import torch.nn as nn


def train(config):
    data_path = config.root_dir + config.data_set_name
    print("Processing Data - ", data_path)
    dataset = KGRAMSData(data_path, config.review_length, mode="train")
    vocab_size = len(dataset.word2idx)
    print("vocab size ", vocab_size)
    user_embedding_idx = dataset.user_id_max
    item_embedding_idx = dataset.item_id_max

    kgrams_model = KGRAMS(word_embedding_size=config.word_embedding_size,
                        vocab_size=vocab_size,
                        out_channels=100,
                        filter_size=3,
                        batch_size=config.batch_size,
                        review_length=config.review_length,
                        user_id_embedding_idx=user_embedding_idx,
                        item_id_embedding_idx=item_embedding_idx,
                        id_embedding_size=config.id_embedding_size,
                        hidden_size=50,
                        latent_size=70,
                        num_of_lstm_layers = 1,
                        num_directions = 1,
                        lstm_hidden_dim = 50)

    data_generator = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, drop_last=True, timeout=0)

    mse_loss = nn.MSELoss()
    crossentr_loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(kgrams_model.parameters(), lr=config.narre_learning_rate)
    # mrg_optimizer = torch.optim.Adam(mrg.parameters(), lr=config.mrg_learning_rate)

    for epoch in range(config.epochs):
        for batch_num, batch in enumerate(data_generator):
            target_user_ids = batch[0]
            target_item_ids = batch[1]
            target_ratings = batch[2]
            target_reviews = torch.stack(batch[3], dim=0).view(config.batch_size, -1)
            user_reviews = batch[4]
            item_reviews = batch[5]
            review_user_ids = torch.stack(batch[6], dim=0).view(config.batch_size, -1)
            review_item_ids = torch.stack(batch[7], dim=0).view(config.batch_size, -1)
            target_reviews_x = torch.stack(batch[8], dim=0).view(config.batch_size, -1)
            target_reviews_y = torch.stack(batch[9], dim=0).view(config.batch_size, -1)

            predicted_rating, review_probabilities = kgrams_model(user_ids=target_user_ids,
                                                              user_reviews=user_reviews,
                                                              user_ids_of_reviews=review_user_ids,
                                                              item_ids=target_item_ids,
                                                              item_reviews=item_reviews,
                                                              item_ids_of_reviews=review_item_ids,
                                                              target_reviews_x = target_reviews_x)
            rating_pred_loss = mse_loss(predicted_rating.squeeze(), target_ratings.float())
            review_probabilities = review_probabilities.view(-1, vocab_size)
            review_gen_loss = crossentr_loss(review_probabilities, target_reviews_y.view(-1))
            print("Rating Loss - ", rating_pred_loss.item(), "LSTM loss - ", review_gen_loss.item())
            optimizer.zero_grad()
            rating_pred_loss.backward(retain_graph = True)
            review_gen_loss.backward()
            optimizer.step()

if __name__ == "__main__":
    config = argparse.ArgumentParser()
    config.add_argument('-root', '--root_dir', required=False, type=str, default="../data/", help='Data root direcroty')
    config.add_argument('-dataset', '--data_set_name', required=False, type=str, default="Musical_Instruments_5.json", help='Dataset')
    config.add_argument('-length', '--review_length', required=False, type=int, default=80,help='Review Length')
    config.add_argument('-batch_size', '--batch_size', required=False, type=int, default=5, help='Batch size')
    config.add_argument('-nlr', '--narre_learning_rate', required=False, type=float, default=0.01, help='NARRE learning rate')
    config.add_argument('-mlr', '--mrg_learning_rate', required=False, type=float, default=0.0003, help='MRG learning rate')
    config.add_argument('-epochs', '--epochs', required=False, type=int, default=100, help='epochs')
    config.add_argument('-wembed', '--word_embedding_size', required=False, type=int, default=100, help='Word embedding size')
    config.add_argument('-iembed', '--id_embedding_size', required=False, type=int, default=100, help='Item/User embedding size')
    config = config.parse_args()
    train(config)
    #"Musical_Instruments_5.json"
    #Digital_Music_5.json