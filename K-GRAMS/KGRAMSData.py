import pickle
from collections import defaultdict
import torch
import copy
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from BaseData import BaseData

PAD_INDEX = 0
PAD_WORD = '<PAD>'

START_INDEX = 1
START_WORD = '<STR>'

END_INDEX = 2
END_WORD = '<END>'

UNK_INDEX = 3
UNK_WORD = '<UNK>'

class KGRAMSData(Dataset, BaseData):
    def __init__(self, data_dir, seq_length, mode = "train"):
        super(KGRAMSData, self).__init__(data_dir)
        #Truncate/Pad review to seq_length
        self.data_set = self.get_mode_data(mode)
        self.data_dict, self.vocab_size, self.user_count, self.item_count = self.preprocess_data(self.data_set, seq_length)
        self.target_user_ids, self.target_item_ids, self.target_ratings, self.target_reviews, self.user_reviews, self.item_reviews, self.review_user_ids, self.review_item_ids = self.load_data()

    def get_mode_data(self, mode):
        if mode is "train":
            data_set = self.train_data
        elif mode is "validate":
            data_set = self.val_data
        else:
            data_set = self.test_data
        return data_set

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
        vocab_size = len(self.word2idx)
        return data_dict, vocab_size, len(set(users)), len(set(items))

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

    def process_data(self, data_dict, user_review_ids, item_review_ids, user_item_data ):
        target_user_ids = []
        target_item_ids = []
        target_ratings = []
        target_reviews = []
        user_reviews = []
        item_reviews = []
        review_user_ids = []
        review_item_ids = []

        for (user_id, item_id, rating, review) in user_item_data:
            one_user_reviews = []
            one_item_reviews = []
            item_ids_of_user = []
            user_ids_of_item = []
            for review_id in user_review_ids[user_id]:
                if data_dict[review_id]["itemID"] is not item_id:
                    one_user_reviews.append(data_dict[review_id]["review"])
                    item_ids_of_user.append(data_dict[review_id]["itemID"])
            for review_id in item_review_ids[item_id]:
                if data_dict[review_id]["userID"] is not user_id:
                    one_item_reviews.append(data_dict[review_id]["review"])
                    user_ids_of_item.append(data_dict[review_id]["userID"])
            if len(one_user_reviews) is not 0 and len(one_item_reviews)is not 0:
                target_user_ids.append(user_id)
                target_item_ids.append(item_id)
                target_ratings.append(rating)
                target_reviews.append(review)
                user_reviews.append(one_user_reviews)
                item_reviews.append(one_item_reviews)
                review_user_ids.append(user_ids_of_item)
                review_item_ids.append(item_ids_of_user)

        self.data_length = len(target_ratings)

        return target_user_ids, target_item_ids, target_ratings, target_reviews, user_reviews, \
               item_reviews, review_user_ids, review_item_ids

    def load_data(self):
        user_review_ids, item_review_ids, user_item_data = self.get_data_dictionaries(self.data_dict)
        return self.process_data(self.data_dict, user_review_ids, item_review_ids, user_item_data)

    def __getitem__(self, idx):
        return self.target_user_ids[idx], self.target_item_ids[idx], self.target_ratings[idx], self.target_reviews[idx], self.user_reviews[idx], self.item_reviews[idx], self.review_user_ids[idx], self.review_item_ids[idx]

    def __len__(self):
        return self.data_length

if __name__ == "__main__":
    root_dir = "../data/"
    data_set_name = "Digital_Music_5.json"
    data_path = root_dir + data_set_name
    print("Processing Data - ", data_path)

    dataset = KGRAMSData(data_path, 80, mode = "train")
    # target_user_ids, target_item_ids, target_ratings, target_reviews, user_reviews, item_reviews, review_user_ids, \
    #                                                                     review_item_ids = data_loader.load_data()

    data_generator = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True, timeout=0)
    #
    # for data in data_generator:
    #     for t in data:
    #         if type(t) is list:
    #             print(len(t), t[0].size())
    #             continue
    #         print(t.size())
    #     # print(data[0].size())
    #     break
