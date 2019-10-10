import pickle
from collections import defaultdict
import torch
import copy

PAD_INDEX = 0
PAD_WORD = '<PAD>'

START_INDEX = 1
START_WORD = '<STR>'

END_INDEX = 2
END_WORD = '<END>'

UNK_INDEX = 3
UNK_WORD = '<UNK>'

class DataLoader():
    def __init__(self, data_dict, seq_length, word2idx, idx2word):
        self.unprocessed_data_dict = data_dict
        self.word2idx = word2idx
        self.idx2word = idx2word
        #Truncate/Pad review to seq_length
        self.data_dict, self.vocab_size, self.user_count, self.item_count = self.preprocess_data(data_dict, seq_length)

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

    def shuffle_data(self, data_dict):
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
                    one_user_reviews.append(torch.tensor(data_dict[review_id]["review"]))
                    item_ids_of_user.append(torch.tensor(data_dict[review_id]["itemID"]))
            for review_id in item_review_ids[item_id]:
                if data_dict[review_id]["userID"] is not user_id:
                    one_item_reviews.append(torch.tensor(data_dict[review_id]["review"]))
                    user_ids_of_item.append(torch.tensor(data_dict[review_id]["userID"]))
            if len(one_user_reviews) is not 0 and len(one_item_reviews)is not 0:
                target_user_ids.append(torch.tensor(user_id))
                target_item_ids.append(torch.tensor(item_id))
                target_ratings.append(torch.tensor(rating))
                target_reviews.append(torch.tensor(review))
                user_reviews.append(torch.stack(one_user_reviews, dim=0).squeeze().view(len(one_user_reviews), -1))
                item_reviews.append(torch.stack(one_item_reviews, dim=0).squeeze().view(len(one_item_reviews), -1))
                review_user_ids.append(torch.stack(user_ids_of_item, dim=0))
                review_item_ids.append(torch.stack(item_ids_of_user, dim=0))

        return target_user_ids, target_item_ids, target_ratings, target_reviews, user_reviews, \
               item_reviews, review_user_ids, review_item_ids

    def load_data(self):
        user_review_ids, item_review_ids, user_item_data = self.shuffle_data(self.data_dict)
        return self.process_data(self.data_dict, user_review_ids, item_review_ids, user_item_data)

if __name__ == "__main__":
    train_data_path = "./data/Digital_Music_5_train.pkl"
    val_data_path = "./data/Digital_Music_5_val.pkl"
    test_data_path = "./data/Digital_Music_5_test.pkl"

    train_data = pickle.load(open(train_data_path, "rb"))
    val_data = pickle.load(open(val_data_path, "rb"))
    test_data = pickle.load(open(test_data_path, "rb"))

    data = {}
    num_of_points = 0
    for rev_id in train_data.keys():
        if num_of_points == 100:
            break
        data[rev_id] = train_data[rev_id]
        num_of_points += 1
    print(len(data))
    data_loader = DataLoader(data, 80)
    target_user_ids, target_item_ids, target_ratings, target_reviews, user_reviews, item_reviews, review_user_ids, \
                                                                        review_item_ids = data_loader.load_data()