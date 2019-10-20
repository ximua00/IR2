import pickle
from collections import defaultdict
import torch
import copy
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from BaseData import BaseData
import numpy as np

PAD_INDEX = 0
PAD_WORD = '<PAD>'

START_INDEX = 1
START_WORD = '<STR>'

END_INDEX = 2
END_WORD = '<END>'

UNK_INDEX = 3
UNK_WORD = '<UNK>'

class KGRAMSData(Dataset, BaseData):
    def __init__(self, data_dir, seq_length, num_reviews_per_user):
        super(KGRAMSData, self).__init__(data_dir)
        self.base_data_set = self.train_data
        self.num_of_reviews = len(self.base_data_set)
        self.num_reviews_per_user = num_reviews_per_user
        self.vocab_size = len(self.word2idx)
        # Truncate/Pad review to seq_length
        self.base_data_dict, self.user_count, self.item_count = self.preprocess_data(self.base_data_set, seq_length)
        self.base_user_review_ids, self.base_item_review_ids, self.base_user_item_data = self.get_data_dictionaries(self.base_data_dict)

    def preprocess_data(self, data_dict_t, seq_length):
        users = []
        items = []
        data_dict = copy.deepcopy(data_dict_t)
        num_of_datapoints = 0
        for review_id in data_dict.keys():
            review = data_dict[review_id]["review"]
            users.append(data_dict[review_id]["userID"])
            items.append(data_dict[review_id]["itemID"])
            review_length = len(review)-2 #excluding <START_WORD> and <END_WORD>
            if review_length > seq_length:
                review_t = review[0:seq_length+1]
            elif review_length < seq_length:
                review_t = review[0:review_length+1]
                for i in range(review_length, seq_length):
                    review_t.append(PAD_INDEX)
            else:
                continue
            review_t.append(END_INDEX)
            data_dict[review_id]["review"] = review_t
        return data_dict, len(set(users)), len(set(items))

    def get_data_dictionaries(self, data_dict):
        user_review_ids = defaultdict(list)
        item_review_ids = defaultdict(list)
        user_item_data = []

        for review_id in data_dict.keys():
            user_id = data_dict[review_id]["userID"]
            item_id = data_dict[review_id]["itemID"]
            review = data_dict[review_id]["review"]
            rating = data_dict[review_id]["rating"]
            user_review_ids[user_id].append(review_id)
            item_review_ids[item_id].append(review_id)
            user_item_data.append((user_id, item_id, rating, review))

        return user_review_ids, item_review_ids, user_item_data

    def process_num_of_reviews(self, review_list):
        while len(review_list) > self.num_reviews_per_user:
            review_list.pop()

    def process_data(self, data_dict, user_review_ids, item_review_ids, user_item_data, mode = "train" ):
        target_user_ids = []
        target_item_ids = []
        target_ratings = []
        target_reviews = []
        target_reviews_x = []
        target_reviews_y = []
        user_reviews = []
        item_reviews = []
        review_user_ids = []
        review_item_ids = []
        max_u_id = 0
        max_i_id = 0
        for (user_id, item_id, rating, review) in user_item_data:
            one_user_reviews = []
            one_item_reviews = []
            item_ids_of_user = []
            user_ids_of_item = []
            if len(user_review_ids[user_id]) > 1 and len(user_review_ids[user_id]) > self.num_reviews_per_user+1:
                self.process_num_of_reviews(user_review_ids[user_id])
                if user_id in user_review_ids.keys():
                    for review_id in user_review_ids[user_id]:
                        if data_dict[review_id]["itemID"] is not item_id:
                            one_user_reviews.append(data_dict[review_id]["review"])
                            item_ids_of_user.append(data_dict[review_id]["itemID"])
                    if max(item_ids_of_user) > max_i_id: max_i_id = max(item_ids_of_user)

            if len(item_review_ids[item_id]) > 1 and len(item_review_ids[item_id]) > self.num_reviews_per_user+1:
                self.process_num_of_reviews(item_review_ids[item_id])
                if item_id in item_review_ids.keys():
                    for review_id in item_review_ids[item_id]:
                        if data_dict[review_id]["userID"] is not user_id:
                            one_item_reviews.append(data_dict[review_id]["review"])
                            user_ids_of_item.append(data_dict[review_id]["userID"])
                    if max(user_ids_of_item) > max_u_id: max_u_id = max(user_ids_of_item)
            if len(one_user_reviews) is not 0 and len(one_item_reviews)is not 0:
                target_user_ids.append(user_id)
                target_item_ids.append(item_id)
                target_ratings.append(rating)
                target_reviews.append(review)
                target_reviews_x.append(review[0:-1])
                target_reviews_y.append(review[1:])
                user_reviews.append(torch.LongTensor(one_user_reviews))
                item_reviews.append(torch.LongTensor(one_item_reviews))
                review_user_ids.append(user_ids_of_item)
                review_item_ids.append(item_ids_of_user)

        self.data_length = len(target_ratings)
        self.user_id_max = max(max(target_user_ids), max_u_id)
        self.item_id_max = max(max(target_item_ids), max_i_id)
        if mode is "train":
            self.user_id_max = max(max(target_user_ids), max_u_id)
            self.item_id_max = max(max(target_item_ids), max_i_id)
        # user_reviews = torch.stack(user_reviews, dim=0)
        # item_reviews = torch.stack(item_reviews, dim=0)
        return target_user_ids, target_item_ids, target_ratings, target_reviews, user_reviews, \
               item_reviews, review_user_ids, review_item_ids, target_reviews_x, target_reviews_y

    def load_data(self):
        return self.process_data(self.base_data_dict, self.base_user_review_ids, self.base_item_review_ids, self.base_user_item_data)

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class KGRAMSTrainData(KGRAMSData):
    def __init__(self, data_dir, seq_length, num_reviews_per_user = 10):
        super(KGRAMSTrainData, self).__init__(data_dir, seq_length, num_reviews_per_user)
        self.target_user_ids, self.target_item_ids, self.target_ratings, self.target_reviews, self.user_reviews, \
        self.item_reviews, self.review_user_ids, self.review_item_ids, self.target_reviews_x, self.target_reviews_y = self.load_data()

    def __getitem__(self, idx):
        return self.target_user_ids[idx], self.target_item_ids[idx], self.target_ratings[idx], self.target_reviews[idx], \
               self.user_reviews[idx], self.item_reviews[idx], self.review_user_ids[idx], self.review_item_ids[idx],\
                self.target_reviews_x[idx], self.target_reviews_y[idx]

    def __len__(self):
        return self.data_length

class KGRAMSEvalData(KGRAMSData):
    def __init__(self, data_dir, seq_length, num_reviews_per_user = 10, mode = "validate"):
        super(KGRAMSEvalData, self).__init__(data_dir, seq_length, num_reviews_per_user)
        self.dataset = self.get_mode_data(mode)
        self.num_of_reviews = len(self.dataset)
        self.eval_data_dict, self.eval_user_count, self.eval_item_count = self.preprocess_data(self.dataset, seq_length)
        self.target_user_ids, self.target_item_ids, self.target_ratings, self.target_reviews, self.user_reviews, \
        self.item_reviews, self.review_user_ids, self.review_item_ids, self.target_reviews_x, self.target_reviews_y = self.load_data()
    def get_mode_data(self, mode):
        if mode is "validate":
            data_set = self.val_data
        else:
            data_set = self.test_data
        return data_set

    def get_eval_data_dictionaries(self, eval_data_dict):
        user_review_ids = defaultdict(list)
        item_review_ids = defaultdict(list)
        user_item_data = []

        for review_id in eval_data_dict.keys():
            user_id = eval_data_dict[review_id]["userID"]
            item_id = eval_data_dict[review_id]["itemID"]
            review = eval_data_dict[review_id]["review"]
            rating = eval_data_dict[review_id]["rating"]
            if user_id in self.base_user_review_ids.keys() and item_id in self.base_item_review_ids.keys():
                user_item_data.append((user_id, item_id, rating, review))

        for (user_id, item_id, rating,review) in user_item_data:
                user_review_ids[user_id] = self.base_user_review_ids[user_id]
                item_review_ids[item_id] = self.base_item_review_ids[item_id]

        return user_review_ids, item_review_ids, user_item_data


    def load_data(self):
        user_review_ids, item_review_ids, user_item_data = self.get_eval_data_dictionaries(self.eval_data_dict)
        return self.process_data(self.base_data_dict, user_review_ids, item_review_ids, user_item_data)

    def __getitem__(self, idx):
        return self.target_user_ids[idx], self.target_item_ids[idx], self.target_ratings[idx], self.target_reviews[idx], \
               self.user_reviews[idx], self.item_reviews[idx], self.review_user_ids[idx], self.review_item_ids[idx],\
                self.target_reviews_x[idx], self.target_reviews_y[idx]

    def __len__(self):
        return self.data_length
