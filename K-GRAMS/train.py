import torch
import argparse
from config import config, device
from KGRAMSData import KGRAMSData, KGRAMSEvalData, KGRAMSTrainData
from model import KGRAMS
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import pickle as pickle
import sys

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
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
    data_generator = DataLoader(dataset, config.batch_size, shuffle=True, num_workers=0, drop_last=True, timeout=0)
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
                                        dataset_object = dataset,
                                        mode = "test")

        # print(word_idx_seq)
        actual_review = get_one_review_from_batch(target_reviews.tolist(), dataset.idx2word)
        generated_review = get_one_review_from_batch(word_idx_seq, dataset.idx2word)
        for (orig, gen) in zip(actual_review, generated_review):
            print("------Original Review----------------")
            print(orig)
            print("--------Generated Review--------------")
            print(gen)

        # generated_review = get_one_review_from_batch(word_idx_seq.tolist(), dataset.idx2word)
        # print("------Original Review----------------")
        # print(actual_review)
        # print("--------Generated Review--------------")
        # print(generated_review)

        # break


def validate(config, model, vocab_size, data_generator, dataset_object):
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
                                                              target_reviews_x=target_reviews_x,
                                                              dataset_object = dataset_object)
        rating_pred_loss = mse_loss(predicted_rating.detach().squeeze(), target_ratings.float())
        review_probabilities = review_probabilities.detach().view(-1, vocab_size)
        review_gen_loss = ce_loss(review_probabilities.detach(), target_reviews_y.view(-1))
        review_batch_loss.append(review_gen_loss.item())
        rating_batch_loss.append(rating_pred_loss.item())
        # break
    return np.mean(rating_batch_loss), np.mean(review_batch_loss)

def train(config):
    data_path = config.root_dir + config.data_set_name
    print("Processing Data - ", data_path)
    dataset_train = KGRAMSTrainData(data_path, config.review_length, num_reviews_per_user=8)
    dataset_val = KGRAMSEvalData(data_path, config.review_length, mode="validate", num_reviews_per_user=8)
    dataset_test = KGRAMSEvalData(data_path, config.review_length, mode="test", num_reviews_per_user=8)

    #### OBTAINING THE GLOVE WORD EMBEDDING MATRIX ####
    glove_embedding_matrix = dataset_train.glove_embedding_matrix
    ##############################################################

    vocab_size = dataset_train.vocab_size
    print("vocab size ", vocab_size)
    user_embedding_idx = dataset_train.user_id_max + 1
    item_embedding_idx = dataset_train.item_id_max + 1

    kgrams_model = KGRAMS(word_embedding_size=config.word_embedding_size,
                        vocab_size=vocab_size,
                        out_channels=100,
                        filter_size=3,
                        batch_size=config.batch_size,
                        review_length=config.review_length,
                        user_id_embedding_idx=user_embedding_idx,
                        item_id_embedding_idx=item_embedding_idx,
                        id_embedding_size=config.id_embedding_size,
                        hidden_size=128,
                        latent_size=256,
                        num_of_lstm_layers = 1,
                        num_directions = 1,
                        lstm_hidden_dim = 128,
                        glove_embedding_matrix = glove_embedding_matrix)
    kgrams_model = kgrams_model.to(device) 

    data_generator_train = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=0, drop_last=True, timeout=0)
    data_generator_val = DataLoader(dataset_val, batch_size=config.batch_size, shuffle=True, num_workers=0, drop_last=True, timeout=0)


    mse_loss = nn.MSELoss()
    crossentr_loss = nn.CrossEntropyLoss()

    # optimizer = torch.optim.Adam(kgrams_model.parameters(), lr=config.narre_learning_rate)
    lstm_modules = []
    base_modules = []
    for m in kgrams_model.parameters():
        if isinstance(m, nn.LSTM):
            lstm_modules.append(m)
        else:
            base_modules.append(m)

    optimizer = torch.optim.Adam([
        {"params": base_modules},
        {"params": lstm_modules, 'lr': config.mrg_learning_rate}
        ],
        lr=config.narre_learning_rate)

    train_rating_loss = []
    train_review_loss = []
    val_rating_loss = []
    val_review_loss = []

    for epoch in range(config.epochs):
        rating_loss = []
        review_loss = []
        # print("Number of training Batches : {}".format(len(data_generator_train)))
        for batch_num, batch in enumerate(data_generator_train):
            # print("Batch Number : {}".format(batch_num))
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
                                                              target_reviews_x = target_reviews_x,
                                                              dataset_object = dataset_train)
            rating_pred_loss = mse_loss(predicted_rating.squeeze(), target_ratings.float())
            review_probabilities = review_probabilities.view(-1, vocab_size)
            review_gen_loss = crossentr_loss(review_probabilities, target_reviews_y.view(-1))
            optimizer.zero_grad()
            rating_pred_loss.backward(retain_graph = True)
            review_gen_loss.backward()
            optimizer.step()
            rating_loss.append(rating_pred_loss.item())
            review_loss.append(review_gen_loss.item())
            # break
        if (epoch % config.eval_freq == 0):
            PATH = "models/model_"+str(epoch)+".pt"
            torch.save(kgrams_model.state_dict(), PATH)
            avg_rating_loss, avg_rev_loss = validate(config, kgrams_model, dataset_val.vocab_size, data_generator_val, dataset_val)
            print("-------------EPOCH:",epoch,"----------------------")
            print("TRAIN : Rating Loss - ", np.mean(rating_loss), "LSTM loss - ", np.mean(review_loss))
            print("VALIDATION:   Rating Loss - ", avg_rating_loss, "LSTM loss - ", avg_rev_loss)
            train_rating_loss.append(np.mean(rating_loss))
            train_review_loss.append((np.mean(review_loss)))
            val_rating_loss.append(avg_rating_loss)
            val_review_loss.append(avg_rev_loss)
            print("--------------Generating review--------------------")
            test(config, kgrams_model, dataset_test)
            print("---------------------------------------------------")
            # break
        # break
    print("Training Completed!")
    dump_pickle("train_rating_loss.pkl", train_rating_loss)
    dump_pickle("train_review_loss.pkl", train_review_loss)
    dump_pickle("val_rating_loss.pkl", val_rating_loss)
    dump_pickle("val_review_loss.pkl", val_review_loss)



    return kgrams_model



if __name__ == "__main__":
    model = train(config)
    # dataset_test = KGRAMSEvalData(data_path, config.review_length, mode="test")
    # config = argparse.ArgumentParser()
    # config.add_argument('-root', '--root_dir', required=False, type=str, default="../data/", help='Data root direcroty')
    # config.add_argument('-dataset', '--data_set_name', required=False, type=str, default="Digital_Music_5.json", help='Dataset')
    # config.add_argument('-length', '--review_length', required=False, type=int, default=80,help='Review Length')
    # config.add_argument('-batch_size', '--batch_size', required=False, type=int, default=5, help='Batch size')
    # config.add_argument('-nlr', '--narre_learning_rate', required=False, type=float, default=0.01, help='NARRE learning rate')
    # config.add_argument('-mlr', '--mrg_learning_rate', required=False, type=float, default=0.0003, help='MRG learning rate')
    # config.add_argument('-epochs', '--epochs', required=False, type=int, default=1, help='epochs')
    # config.add_argument('-wembed', '--word_embedding_size', required=False, type=int, default= 1024, help='Word embedding size')
    # config.add_argument('-iembed', '--id_embedding_size', required=False, type=int, default=100, help='Item/User embedding size')
    # config.add_argument('-efreq', '--eval_freq', required=False, type=int, default=100,help='Evaluation Frequency')
    # config = config.parse_args()
    # model = train(config)
    # data_path = config.root_dir + config.data_set_name
    # dataset_test = KGRAMSEvalData(data_path, config.review_length, mode="test")
    # test(config, model, dataset_test)
    #"Musical_Instruments_5.json"
    #Digital_Music_5.json