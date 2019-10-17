import json
from collections import defaultdict, Counter 
from pprint import pprint
import numpy as np
import re
import pickle

np.random.seed(2017)

PAD_INDEX = 0
PAD_WORD = '<PAD>'

START_INDEX = 1
START_WORD = '<STR>'

END_INDEX = 2
END_WORD = '<END>'

UNK_INDEX = 3
UNK_WORD = '<UNK>'


class BaseData:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.word2idx = {}
        self.unk_frequency = 5

        self.text_data = []

        self.index_counter = 3

        self.user_idx = {}
        self.user_idx_counter = 0

        self.item_idx = {}
        self.item_idx_counter = 0

        self.data = self.parse_data()
        self.train_data, self.val_data, self.test_data = self.split_data()
        self.word2idx, self.idx2word = self.make_mappings(self.train_data)
        self.generate_review_idx(self.val_data)
        self.generate_review_idx(self.test_data)

    def parse_data(self):
        data = defaultdict(dict)
        with open(self.data_dir, 'r') as jsonfile:
            for reviewID, line in enumerate((jsonfile)):
                line_data = json.loads(line)
                if line_data["reviewerID"] in self.user_idx.keys():
                    data[reviewID]["userID"] = self.user_idx[line_data["reviewerID"]]
                else:
                    self.user_idx_counter += 1
                    data[reviewID]["userID"] = self.user_idx_counter
                    self.user_idx[line_data["reviewerID"]] = self.user_idx_counter

                if line_data["asin"] in self.item_idx.keys():
                    data[reviewID]["itemID"] = self.item_idx[line_data["asin"]]
                else:
                    self.item_idx_counter += 1
                    data[reviewID]["itemID"] = self.item_idx_counter
                    self.item_idx[line_data["asin"]] = self.item_idx_counter

                data[reviewID]["helpful"] = line_data["helpful"]
                data[reviewID]["rating"] = line_data["overall"]
                data[reviewID]["reviewText"] = line_data["reviewText"]
                # file_name.write(str(reviewID) +"  " + line_data["reviewText"]+ "\n")
                line_data["reviewText"] = line_data["reviewText"].lower()
                text_split = re.split("[ .,;()&!]+", line_data["reviewText"])
                data[reviewID]["reviewSplitText"] = text_split

        return data

    def make_mappings(self,data):
        word2idx = {PAD_WORD:PAD_INDEX, START_WORD:START_INDEX, END_WORD:END_INDEX, UNK_WORD:UNK_INDEX}
        idx2word = {PAD_INDEX:PAD_WORD, START_INDEX:START_WORD, END_INDEX:END_WORD, UNK_INDEX:UNK_WORD}

        word_counter = Counter(self.text_data)
        vocab_file = open("vocab.txt", "w")
        for review_id in data.keys():
            review_sentence_idxs = [START_INDEX]
            for word in data[review_id]["reviewSplitText"]:
                if word_counter[word] > self.unk_frequency:
                    # Add to the vocabulary
                    if word not in word2idx.keys():
                        self.index_counter += 1
                        word2idx[word] = self.index_counter
                        idx2word[self.index_counter] = word
                        vocab_file.write(str(self.index_counter) + " - " + word + "\n")
                        # Add word idx to review
                        review_sentence_idxs.append(self.index_counter)
                    else:
                        review_sentence_idxs.append((word2idx[word]))
                else:
                    review_sentence_idxs.append(UNK_INDEX)

            review_sentence_idxs.append(END_INDEX)
            data[review_id]['review'] = review_sentence_idxs
        return word2idx, idx2word

    def generate_review_idx(self, data):
        for review_id in data.keys():
            review_sentence_idxs = [START_INDEX]
            for word in data[review_id]["reviewSplitText"]:
                if word in self.word2idx.keys():
                    review_sentence_idxs.append(self.word2idx[word])
                else:
                    review_sentence_idxs.append(UNK_INDEX)
            review_sentence_idxs.append(END_INDEX)
            data[review_id]['review'] = review_sentence_idxs


    def split_data(self, train_split = 0.7, val_split = 0.15, test_split = 0.15):
        train_set = {}
        val_set = {}
        test_set = {}
        data_size = len(self.data)
        train_len = int(train_split * data_size)
        val_len = int(val_split * data_size)
        test_len = int(test_split * data_size)
        review_ids = [i for i in range(0, data_size)]
        review_ids = np.random.permutation(review_ids)
        #Train set
        for i in range(train_len):
            train_set[review_ids[i]] = self.data[review_ids[i]]
            [self.text_data.append(word) for word in self.data[review_ids[i]]['reviewSplitText']]
        #Validation set
        for i in range(train_len, train_len+val_len):
            val_set[review_ids[i]] = self.data[review_ids[i]]
        #Test set
        for i in range(train_len+val_len, data_size):
            test_set[review_ids[i]] = self.data[review_ids[i]]

        return train_set, val_set, test_set

    def dump_pickle(self, root_dir, data_set_name):
        data_set = data_set_name.split(".")[0]
        train_pkl = root_dir + data_set + "_train.pkl"
        val_pkl = root_dir + data_set + "_val.pkl"
        test_pkl = root_dir + data_set + "_test.pkl"
        word2idx_pkl = root_dir + data_set + "_word2idx.pkl"
        idx2word_pkl = root_dir + data_set + "_idx2word.pkl"
        #Train data
        pickle_file = open(train_pkl, 'wb')
        print("Dumping pickle : ", train_pkl)
        pickle.dump(self.train_data, pickle_file)
        #Validation data
        pickle_file = open(val_pkl, 'wb')
        print("Dumping pickle : ", val_pkl)
        pickle.dump(self.val_data, pickle_file)
        #Test data
        pickle_file = open(test_pkl, 'wb')
        print("Dumping pickle : ", test_pkl)
        pickle.dump(self.test_data, pickle_file)
        # word2idx data
        pickle_file = open(word2idx_pkl, 'wb')
        print("Dumping pickle : ", word2idx_pkl)
        pickle.dump(self.word2idx, pickle_file)
        # idx2word data
        pickle_file = open(idx2word_pkl, 'wb')
        print("Dumping pickle : ", idx2word_pkl)
        pickle.dump(self.idx2word, pickle_file)

#
if __name__ == "__main__":
    root_dir = "./data/"
    data_set_name = "Digital_Music_5.json"
    data_path = root_dir+data_set_name
    print("Processing Data - ", data_path)
    base_data = BaseData(data_path)
    # base_data.dump_pickle(root_dir, data_set_name)

    # basedata is an instance of the BaseData class
    # base_data.train_data/val_data/test_data outputs a defaultdictionary with review_ids as key.
    # each value contains {["userID"]:X, ["itemID"]:X, ["helpful"]:X, ["rating"]:X, ["review"]:X}
    # userID is an index
    # itemID is an index
    # helpful is a list of [X,X]
    # rating is a number from 0-5
    # review is a list of indices corresponding to each word. Already with START and END of review.
            # Thresholded at min 3 words, lowercased, NOT PADDED.

