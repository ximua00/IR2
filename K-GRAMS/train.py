import torch
import argparse
from KGRAMSData import KGRAMSData, KGRAMSEvalData, KGRAMSTrainData
from model import KGRAMS
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def test(config, model, dataset):
    data_generator = DataLoader(dataset, config.batch_size, shuffle=True, num_workers=0,drop_last=True, timeout=0)
    for input in data_generator:
        target_user_id = input[0].to(device)
        target_item_id = input[1].to(device)
        user_reviews = input[4].to(device)
        item_reviews = input[5].to(device)
        review_user_ids = torch.stack(input[6], dim=0).view(config.batch_size, -1).to(device)
        review_item_ids = torch.stack(input[7], dim=0).view(config.batch_size, -1).to(device)
        rating_pred, word_idx_seq = model(user_ids=target_user_id,
                                        user_reviews=user_reviews,
                                        user_ids_of_reviews=review_user_ids,
                                        item_ids=target_item_id,
                                        item_reviews=item_reviews,
                                        item_ids_of_reviews=review_item_ids,
                                        target_reviews_x=None,
                                        mode = "test")

        # print(word_idx_seq)

        wordidx_batch = word_idx_seq.tolist()
        for batch in wordidx_batch:
            review = []
            for idx in batch:
                review.append(dataset.idx2word[idx])
            print(" ".join(review))
            break
        break


def validate(config, model, vocab_size, data_generator):
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    rating_batch_loss = []
    review_batch_loss = []
    for batch_num, batch in enumerate(data_generator):
        target_user_ids = batch[0].to(device)
        target_item_ids = batch[1].to(device)
        target_ratings = batch[2].to(device)
        target_reviews = torch.stack(batch[3], dim=0).view(config.batch_size, -1).to(device)
        user_reviews = batch[4].to(device)
        item_reviews = batch[5].to(device)
        review_user_ids = torch.stack(batch[6], dim=0).view(config.batch_size, -1).to(device)
        review_item_ids = torch.stack(batch[7], dim=0).view(config.batch_size, -1).to(device)
        target_reviews_x = torch.stack(batch[8], dim=0).view(config.batch_size, -1).to(device)
        target_reviews_y = torch.stack(batch[9], dim=0).view(config.batch_size, -1).to(device)

        predicted_rating, review_probabilities = model(user_ids=target_user_ids,
                                                              user_reviews=user_reviews,
                                                              user_ids_of_reviews=review_user_ids,
                                                              item_ids=target_item_ids,
                                                              item_reviews=item_reviews,
                                                              item_ids_of_reviews=review_item_ids,
                                                              target_reviews_x=target_reviews_x)
        rating_pred_loss = mse_loss(predicted_rating.detach().squeeze(), target_ratings.float())
        review_probabilities = review_probabilities.detach().view(-1, vocab_size)
        review_gen_loss = ce_loss(review_probabilities.detach(), target_reviews_y.view(-1))
        review_batch_loss.append(review_gen_loss.item())
        rating_batch_loss.append(rating_pred_loss.item())
    return np.mean(rating_batch_loss), np.mean(review_batch_loss)

def train(config):
    data_path = config.root_dir + config.data_set_name
    print("Processing Data - ", data_path)
    dataset_train = KGRAMSTrainData(data_path, config.review_length)
    dataset_val = KGRAMSEvalData(data_path, config.review_length, mode="validate")
    dataset_test = KGRAMSEvalData(data_path, config.review_length, mode="test")
    vocab_size = dataset_train.vocab_size
    print("vocab size ", vocab_size)
    user_embedding_idx = dataset_train.user_id_max
    item_embedding_idx = dataset_train.item_id_max

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
    kgrams_model = kgrams_model.to(device)

    data_generator_train = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=0, drop_last=True, timeout=0)
    data_generator_val = DataLoader(dataset_val, batch_size=config.batch_size, shuffle=True, num_workers=0, drop_last=True, timeout=0)

    mse_loss = nn.MSELoss()
    crossentr_loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(kgrams_model.parameters(), lr=config.narre_learning_rate)
    # mrg_optimizer = torch.optim.Adam(mrg.parameters(), lr=config.mrg_learning_rate)

    for epoch in range(config.epochs):
        rating_loss = []
        review_loss = []
        for batch_num, batch in enumerate(data_generator_train):
            target_user_ids = batch[0].to(device)
            target_item_ids = batch[1].to(device)
            target_ratings = batch[2].to(device)
            target_reviews = torch.stack(batch[3], dim=0).view(config.batch_size, -1).to(device)
            user_reviews = batch[4].to(device)
            item_reviews = batch[5].to(device)
            review_user_ids = torch.stack(batch[6], dim=0).view(config.batch_size, -1).to(device)
            review_item_ids = torch.stack(batch[7], dim=0).view(config.batch_size, -1).to(device)
            target_reviews_x = torch.stack(batch[8], dim=0).view(config.batch_size, -1).to(device)
            target_reviews_y = torch.stack(batch[9], dim=0).view(config.batch_size, -1).to(device)

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
            optimizer.zero_grad()
            rating_pred_loss.backward(retain_graph = True)
            review_gen_loss.backward()
            optimizer.step()
            rating_loss.append(rating_pred_loss.item())
            review_loss.append(review_gen_loss.item())
        if (epoch % config.eval_freq == 0):
            avg_rating_loss, avg_rev_loss = validate(config, kgrams_model, dataset_val.vocab_size, data_generator_val)
            print("TRAIN : Rating Loss - ", np.mean(rating_loss), "LSTM loss - ", np.mean(review_loss))
            print("VALIDATION:   Rating Loss - ", avg_rating_loss, "LSTM loss - ", avg_rev_loss)
            print("--------------Generating review--------------------")
            test(config, kgrams_model, dataset_test)
            print("---------------------------------------------------")

    print("Training Completed!")
    return kgrams_model



if __name__ == "__main__":
    config = argparse.ArgumentParser()
    config.add_argument('-root', '--root_dir', required=False, type=str, default="../data/", help='Data root direcroty')
    config.add_argument('-dataset', '--data_set_name', required=False, type=str, default="Digital_Music_5.json", help='Dataset')
    config.add_argument('-length', '--review_length', required=False, type=int, default=80,help='Review Length')
    config.add_argument('-batch_size', '--batch_size', required=False, type=int, default=8, help='Batch size')
    config.add_argument('-nlr', '--narre_learning_rate', required=False, type=float, default=0.01, help='NARRE learning rate')
    config.add_argument('-mlr', '--mrg_learning_rate', required=False, type=float, default=0.0003, help='MRG learning rate')
    config.add_argument('-epochs', '--epochs', required=False, type=int, default=100, help='epochs')
    config.add_argument('-wembed', '--word_embedding_size', required=False, type=int, default=100, help='Word embedding size')
    config.add_argument('-iembed', '--id_embedding_size', required=False, type=int, default=100, help='Item/User embedding size')
    config.add_argument('-efreq', '--eval_freq', required=False, type=int, default=10,help='Evaluation Frequency')
    config = config.parse_args()
    model = train(config)
    # test(config, model)
    #"Musical_Instruments_5.json"
    #Digital_Music_5.json