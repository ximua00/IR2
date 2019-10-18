import pickle
import hickle
from BaseData import BaseData

class MRGData(BaseData):
    def __init__(self, data_dir):
        super().__init__(data_dir)
        self.output_pickle()

    def output_pickle(self):
        #Review_id, User, item, review_test, photo_id
        self.output_review_pkl(pickle_file_name = "train.pkl", data_set = self.train_data)
        self.output_review_pkl(pickle_file_name= "val.pkl", data_set=self.val_data)
        self.output_review_pkl(pickle_file_name= "test.pkl", data_set=self.test_data)

        self.output_vocab_pkl(pickle_file_name= "vocab.pkl", data_set=self.word2idx)

        self.output_counter_pickle(pickle_file_name= "users.pkl", count=self.user_idx_counter)
        self.output_counter_pickle(pickle_file_name="items.pkl", count=self.item_idx_counter)

    def output_counter_pickle(self, pickle_file_name, count):
        pickle_file = open(pickle_file_name, 'wb')
        pickle.dump(count, pickle_file)

    def output_review_pkl(self, pickle_file_name, data_set):
        dummy_image_id = '--3_rzdtCKQSq7Hpdhh_5Q'
        pickle_file = open(pickle_file_name, 'wb')
        for review in data_set.keys():
            data_point = {}
            data_point['_id'] = review
            data_point['Item'] = data_set[review]['itemID']
            data_point['Rating'] = data_set[review]['rating']
            data_point['Reviews'] ={dummy_image_id:[data_set[review]['review']]}
            data_point['User'] = data_set[review]['userID']
            data_point['Photos'] = dummy_image_id
            pickle.dump(data_point, pickle_file)

    def output_vocab_pkl(self, pickle_file_name, data_set):
        pickle_file = open(pickle_file_name, 'wb')
        pickle.dump(data_set, pickle_file)

if __name__ == "__main__":
    mrg_data = MRGData("./data/Digital_Music_5.json")
    # train_pickle = pickle.load(open("train.pkl", "rb"))
    # with open("train.pkl", 'rb') as f:
    #   try:
    #     while True:
    #       exp = pickle.load(f)
    #       print(exp)
    #   except EOFError:
    #     pass
