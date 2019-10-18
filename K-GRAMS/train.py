import torch
import argparse
from config import config, device
from KGRAMSData import KGRAMSData, KGRAMSEvalData, KGRAMSTrainData
from model import KGRAMS
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import pickle

np.random.seed(2017)

def dump_pickle(file_name,data):
    pickle_file = open(file_name, 'wb')
    print("Dumping pickle : ", file_name)
    pickle.dump(data, pickle_file)

def get_one_review_from_batch(word_idx_batch_list, idx2word):
    batch_reviews = []
    for batch in word_idx_batch_list:
        review = []
        for idx in batch:
            review.append(idx2word[idx])
        review = (" ".join(review))
        batch_reviews.append(review)
    return batch_reviews


def test(config, model, dataset):
    data_generator = DataLoader(dataset, config.batch_size, shuffle=True, num_workers=0,drop_last=True, timeout=0)
    for input in data_generator:
        target_user_id = input[0].to(device)
        target_item_id = input[1].to(device)
        user_reviews = input[4].to(device)
        item_reviews = input[5].to(device)
        target_reviews = torch.stack(input[3], dim=1).view(config.batch_size, -1).to(device)
        review_user_ids = torch.stack(input[6], dim=1).view(config.batch_size, -1).to(device)
        review_item_ids = torch.stack(input[7], dim=1).view(config.batch_size, -1).to(device)
        rating_pred, word_idx_seq = model(user_ids=target_user_id,
                                        user_reviews=user_reviews,
                                        user_ids_of_reviews=review_user_ids,
                                        item_ids=target_item_id,
                                        item_reviews=item_reviews,
                                        item_ids_of_reviews=review_item_ids,
                                        target_reviews_x=None,
                                        mode = "test")

        actual_review = get_one_review_from_batch(target_reviews.tolist(), dataset.idx2word)
        generated_review = get_one_review_from_batch(word_idx_seq, dataset.idx2word)
        for (orig, gen) in zip(actual_review, generated_review):
            print("------Original Review----------------")
            print(orig)
            print("--------Generated Review--------------")
            print(gen)

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
        target_reviews = torch.stack(batch[3], dim=1).view(config.batch_size, -1).to(device)
        user_reviews = batch[4].to(device)
        item_reviews = batch[5].to(device)
        review_user_ids = torch.stack(batch[6], dim=1).view(config.batch_size, -1).to(device)
        review_item_ids = torch.stack(batch[7], dim=1).view(config.batch_size, -1).to(device)
        target_reviews_x = torch.stack(batch[8], dim=1).view(config.batch_size, -1).to(device)
        target_reviews_y = torch.stack(batch[9], dim=1).view(config.batch_size, -1).to(device)

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
    dataset_train = KGRAMSTrainData(data_path, config.review_length, num_reviews_per_user = 8)
    dataset_val = KGRAMSEvalData(data_path, config.review_length, mode="validate", num_reviews_per_user=8)
    # dataset_test = KGRAMSEvalData(data_path, config.review_length, mode="test", num_reviews_per_user=8)

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

    train_rating_loss = []
    train_review_loss = []
    val_rating_loss = []
    val_review_loss = []

    for epoch in range(config.epochs):
        rating_loss = []
        review_loss = []
        for batch_num, batch in enumerate(data_generator_train):
            target_user_ids = batch[0].to(device)
            target_item_ids = batch[1].to(device)
            target_ratings = batch[2].to(device)
            target_reviews = torch.stack(batch[3], dim=1).view(config.batch_size, -1).to(device)
            user_reviews = batch[4].to(device)
            item_reviews = batch[5].to(device)
            review_user_ids = torch.stack(batch[6], dim=1).view(config.batch_size, -1).to(device)
            review_item_ids = torch.stack(batch[7], dim=1).view(config.batch_size, -1).to(device)
            target_reviews_x = torch.stack(batch[8], dim=1).view(config.batch_size, -1).to(device)
            target_reviews_y = torch.stack(batch[9], dim=1).view(config.batch_size, -1).to(device)

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
            PATH = "models/model_"+str(epoch)+".pt"
            torch.save(kgrams_model, PATH)
            avg_rating_loss, avg_rev_loss = validate(config, kgrams_model, dataset_train.vocab_size, data_generator_val)
            print("-------------EPOCH:",epoch,"----------------------")
            print("TRAIN : Rating Loss - ", np.mean(rating_loss), "LSTM loss - ", np.mean(review_loss))
            print("VALIDATION:   Rating Loss - ", avg_rating_loss, "LSTM loss - ", avg_rev_loss)
            train_rating_loss.append(np.mean(rating_loss))
            train_review_loss.append((np.mean(review_loss)))
            val_rating_loss.append(avg_rating_loss)
            val_review_loss.append(avg_rev_loss)
            # print("--------------Generating review--------------------")
            # test(config, kgrams_model, dataset_test)
            # print("---------------------------------------------------")

    print("Training Completed!")
    dump_pickle("train_rating_loss.pkl", train_rating_loss)
    dump_pickle("train_review_loss.pkl", train_review_loss)
    dump_pickle("val_rating_loss.pkl", val_rating_loss)
    dump_pickle("val_review_loss.pkl", val_review_loss)

    return kgrams_model



if __name__ == "__main__":
    model = train(config)