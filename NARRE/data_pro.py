import numpy as np
import re
import itertools
from collections import Counter

import tensorflow as tf
import csv
import pickle
import os

import sys

tf.flags.DEFINE_string("valid_data", "/content/NARRE/data/music/music_valid.csv", " Data for validation") # Flag for path to validation CSV file
tf.flags.DEFINE_string("test_data", "/content/NARRE/data/music/music_test.csv", "Data for testing")# Flag for path to test CSV file
tf.flags.DEFINE_string("train_data", "/content/NARRE/data/music/music_train.csv", "Data for training")# Flag for path to train CSV file
tf.flags.DEFINE_string("user_review", "/content/NARRE/data/music/user_review", "User's reviews") # Flag for path to the user_review pickle file
tf.flags.DEFINE_string("item_review", "/content/NARRE/data/music/item_review", "Item's reviews") # Flag for path to the item_review pickle file
tf.flags.DEFINE_string("user_review_id", "/content/NARRE/data/music/user_rid", "user_review_id") # Flag for path to the user_id pickle file
tf.flags.DEFINE_string("item_review_id", "/content/NARRE/data/music/item_rid", "item_review_id") # Flag for path to the item_id pickle file
tf.flags.DEFINE_string("stopwords", "/content/NARRE/data/stopwords", "stopwords") # Flag for path to the stopwords pickle file


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z]", " ", string) # Replace non-alphabets with space
    string = re.sub(r"\'s", " \'s", string) # Replace 's with ^'s (^ stands for space)
    string = re.sub(r"\'ve", " \'ve", string) # Replace 've with ^'ve (^ stands for space)
    string = re.sub(r"n\'t", " n\'t", string) # Replace n't with ^n't (^ stands for space)
    string = re.sub(r"\'re", " \'re", string) # Replace 're with ^'re (^ stands for space)
    string = re.sub(r"\'d", " \'d", string) # Replace 'd with ^'d (^ stands for space)
    string = re.sub(r"\'ll", " \'ll", string) # Replace 'll with ^'ll (^ stands for space)
    string = re.sub(r",", " , ", string) # Replace , with ^,^ (^ stands for space)
    string = re.sub(r"!", " ! ", string) # Replace ! with ^!^ (^ stands for space)
    string = re.sub(r"\(", " \( ", string) # Replace ( with ^(^ (^ stands for space)
    string = re.sub(r"\)", " \) ", string) # Replace ) with ^)^ (^ stands for space)
    string = re.sub(r"\?", " \? ", string) # Replace ? with ^?^ (^ stands for space)
    string = re.sub(r"\s{2,}", " ", string) # Replace more than two white spaces with one space
    return string.strip().lower() # Removes beginning and trailing white space characters and converts to lowercase


def pad_sentences(u_text, u_len, u2_len, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    review_num = u_len # The maximum number of reviews
    review_len = u2_len # The maximum review length

    u_text2 = {}
    for i in u_text.keys(): # For each user ID
        u_reviews = u_text[i] # List of Lists; each sub list is a tokenized review written by that particular user ID
        padded_u_train = []
        for ri in range(review_num): # For each amount of reviews written
            if ri < len(u_reviews): # If less than the number of reviews written by that user
                sentence = u_reviews[ri] # Access a particular tokenized review [List]
                if review_len > len(sentence): # If the maximum review length is greater than the target review length
                    num_padding = review_len - len(sentence) # Get the amount to pad
                    new_sentence = sentence + [padding_word] * num_padding # Pad the sentence with the padding token
                    padded_u_train.append(new_sentence) # Add to the list of padded sentences
                else: # If the maximum review length is smaller than the target review length
                    new_sentence = sentence[:review_len] # Trim the sentence up to the maximum review length
                    padded_u_train.append(new_sentence) # Add to the list of padded sentences
            else: # If greater than the amount of reviews written by that user
                new_sentence = [padding_word] * review_len # Dummy sentence consisting of pad tokens
                padded_u_train.append(new_sentence) # Add to the list of padded sentences
        u_text2[i] = padded_u_train # List of padded sentences corresponding to the user ID

    return u_text2

# Keep the number of reviews to the max number of reviews
def pad_reviewid(u_train, u_valid, u_len, num):
    pad_u_train = []

    for i in range(len(u_train)):
        x = u_train[i] # List of Item IDs pertaining to user at index i
        while u_len > len(x): # If max num of reviews is greater than the number of reviews written by user i
            x.append(num) # Pad the difference using the total number of items
        if u_len < len(x): # If max num of reviews is lesser than the number of reviews written by user i
            x = x[:u_len]     # Truncate to the smallest length
        pad_u_train.append(x) 
    pad_u_valid = []

    for i in range(len(u_valid)):
        x = u_valid[i]
        while u_len > len(x):
            x.append(num)
        if u_len < len(x):
            x = x[:u_len]
        pad_u_valid.append(x)
    return pad_u_train, pad_u_valid

def build_vocab(sentences1, sentences2):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    
    # Build vocabulary
    word_counts1 = Counter(itertools.chain(*sentences1))
    # Mapping from index to word
    vocabulary_inv1 = [x[0] for x in word_counts1.most_common()] # List of words in descending order
    vocabulary_inv1 = list(sorted(vocabulary_inv1)) # Alphabetical sorting
    # Mapping from word to index
    vocabulary1 = {x: i for i, x in enumerate(vocabulary_inv1)}

    word_counts2 = Counter(itertools.chain(*sentences2))
    # Mapping from index to word
    vocabulary_inv2 = [x[0] for x in word_counts2.most_common()]
    vocabulary_inv2 = list(sorted(vocabulary_inv2))
    # Mapping from word to index
    vocabulary2 = {x: i for i, x in enumerate(vocabulary_inv2)}
    return [vocabulary1, vocabulary_inv1, vocabulary2, vocabulary_inv2]


def build_input_data(u_text, i_text, vocabulary_u, vocabulary_i):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    l = len(u_text) # Number of users
    u_text2 = {}
    for i in u_text.keys():
        u_reviews = u_text[i] # List Of Lists (tokenized reviews)
        u = np.array([[vocabulary_u[word] for word in words] for words in u_reviews]) # List of lists [[idx1, idx2 ... ], [idx1,idx2], ..]
        u_text2[i] = u
    l = len(i_text) # Number of items
    i_text2 = {}
    for j in i_text.keys():
        i_reviews = i_text[j]
        i = np.array([[vocabulary_i[word] for word in words] for words in i_reviews])
        i_text2[j] = i
    return u_text2, i_text2


def load_data(train_data, valid_data, user_review, item_review, user_rid, item_rid, stopwords):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    u_text, i_text, y_train, y_valid, u_len, i_len, u2_len, i2_len, uid_train, iid_train, uid_valid, iid_valid, user_num, item_num \
        , reid_user_train, reid_item_train, reid_user_valid, reid_item_valid = \
        load_data_and_labels(train_data, valid_data, user_review, item_review, user_rid, item_rid, stopwords)
    print "load data done"
    u_text = pad_sentences(u_text, u_len, u2_len)
    reid_user_train, reid_user_valid = pad_reviewid(reid_user_train, reid_user_valid, u_len, item_num + 1)

    print "pad user done"
    i_text = pad_sentences(i_text, i_len, i2_len)
    reid_item_train, reid_item_valid = pad_reviewid(reid_item_train, reid_item_valid, i_len, user_num + 1)

    print "pad item done" 

    
    user_voc = [xx for x in u_text.itervalues() for xx in x] # List of Lists; each sub list is a tokenized review
    item_voc = [xx for x in i_text.itervalues() for xx in x] # List of Lists; each sub list is a tokenized review

    vocabulary_user, vocabulary_inv_user, vocabulary_item, vocabulary_inv_item = build_vocab(user_voc, item_voc)
    print len(vocabulary_user) # Length of user dictionary
    print len(vocabulary_item) # Length of item dictionary
    u_text, i_text = build_input_data(u_text, i_text, vocabulary_user, vocabulary_item)
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    uid_train = np.array(uid_train)
    uid_valid = np.array(uid_valid)
    iid_train = np.array(iid_train)
    iid_valid = np.array(iid_valid)
    reid_user_train = np.array(reid_user_train)
    reid_user_valid = np.array(reid_user_valid)
    reid_item_train = np.array(reid_item_train)
    reid_item_valid = np.array(reid_item_valid)

    return [u_text, i_text, y_train, y_valid, vocabulary_user, vocabulary_inv_user, vocabulary_item,
            vocabulary_inv_item, uid_train, iid_train, uid_valid, iid_valid, user_num, item_num, reid_user_train,
            reid_item_train, reid_user_valid, reid_item_valid]


def load_data_and_labels(train_data, valid_data, user_review, item_review, user_rid, item_rid, stopwords):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files


    f_train = open(train_data, "r") 
    f1 = open(user_review)
    f2 = open(item_review)
    f3 = open(user_rid)
    f4 = open(item_rid)

    user_reviews = pickle.load(f1)
    item_reviews = pickle.load(f2)
    user_rids = pickle.load(f3)
    item_rids = pickle.load(f4)

    reid_user_train = [] # List of lists; each sub list is a list of item ids corresponding to the user id at a particular index
    reid_item_train = []
    uid_train = []
    iid_train = []
    y_train = []
    u_text = {} # {User ID: [[tokenized_review_1], [tokenized_review_2],...]}
    u_rid = {} # {User ID: [item_ids]}
    i_text = {} # {Item ID: [[tokenized_review_1], [tokenized_review_2], ...]}
    i_rid = {} # {Item ID: [user_ids]}
    i = 0
    for line in f_train:
        i = i + 1
        line = line.split(',')
        uid_train.append(int(line[0])) # appending user id
        iid_train.append(int(line[1])) # appending item id
        if u_text.has_key(int(line[0])): 
            reid_user_train.append(u_rid[int(line[0])])
        else:
            u_text[int(line[0])] = [] 
            for s in user_reviews[int(line[0])]:
                s1 = clean_str(s)
                s1 = s1.split(" ")
                u_text[int(line[0])].append(s1)
            u_rid[int(line[0])] = []
            for s in user_rids[int(line[0])]:
                u_rid[int(line[0])].append(int(s))
            reid_user_train.append(u_rid[int(line[0])])

        if i_text.has_key(int(line[1])):
            reid_item_train.append(i_rid[int(line[1])])
        else:
            i_text[int(line[1])] = []
            for s in item_reviews[int(line[1])]:
                s1 = clean_str(s)
                s1 = s1.split(" ")

                i_text[int(line[1])].append(s1)
            i_rid[int(line[1])] = []
            for s in item_rids[int(line[1])]:
                i_rid[int(line[1])].append(int(s))
            reid_item_train.append(i_rid[int(line[1])])
        y_train.append(float(line[2]))
    print "valid"
    reid_user_valid = []
    reid_item_valid = []

    uid_valid = []
    iid_valid = []
    y_valid = []
    f_valid = open(valid_data)
    for line in f_valid:
        line = line.split(',')
        uid_valid.append(int(line[0]))
        iid_valid.append(int(line[1]))
        if u_text.has_key(int(line[0])):
            reid_user_valid.append(u_rid[int(line[0])])
        else:
            u_text[int(line[0])] = [['<PAD/>']]
            u_rid[int(line[0])] = [int(0)]
            reid_user_valid.append(u_rid[int(line[0])])

        if i_text.has_key(int(line[1])):
            reid_item_valid.append(i_rid[int(line[1])])
        else:
            i_text[int(line[1])] = [['<PAD/>']]
            i_rid[int(line[1])] = [int(0)]
            reid_item_valid.append(i_rid[int(line[1])])

        y_valid.append(float(line[2]))
    print "len"


    review_num_u = np.array([len(x) for x in u_text.itervalues()]) # [review_num_1, review_num_2,...] No of reviews per user
    x = np.sort(review_num_u) # Sorting ascending
    u_len = x[int(0.9 * len(review_num_u)) - 1] # The maximum number of reviews that can be written
    review_len_u = np.array([len(j) for i in u_text.itervalues() for j in i]) # [review_1_user_1_len, review_2_user_1_len, ..]
    x2 = np.sort(review_len_u) # Sorting Ascending
    u2_len = x2[int(0.9 * len(review_len_u)) - 1] # The maximum review length

    review_num_i = np.array([len(x) for x in i_text.itervalues()]) 
    y = np.sort(review_num_i) 
    i_len = y[int(0.9 * len(review_num_i)) - 1] 
    review_len_i = np.array([len(j) for i in i_text.itervalues() for j in i])
    y2 = np.sort(review_len_i) 
    i2_len = y2[int(0.9 * len(review_len_i)) - 1]
    print "u_len:", u_len # The first k users based on the number of reviews they've written
    print "i_len:", i_len # The first k items based on the number of reviews they have
    print "u2_len:", u2_len # The first k user reviews based on their lengths
    print "i2_len:", i2_len # The first k item reviews based on their lengths
    user_num = len(u_text) # Number of unique user IDs
    item_num = len(i_text) # Number of unique text IDS
    print "user_num:", user_num
    print "item_num:", item_num
    return [u_text, i_text, y_train, y_valid, u_len, i_len, u2_len, i2_len, uid_train,
            iid_train, uid_valid, iid_valid, user_num,
            item_num, reid_user_train, reid_item_train, reid_user_valid, reid_item_valid]


if __name__ == '__main__':
    TPS_DIR = '/content/NARRE/data/music'
    FLAGS = tf.flags.FLAGS
    #FLAGS._parse_flags()
    FLAGS(sys.argv)

    u_text, i_text, y_train, y_valid, vocabulary_user, vocabulary_inv_user, vocabulary_item, \
    vocabulary_inv_item, uid_train, iid_train, uid_valid, iid_valid, user_num, item_num, reid_user_train, reid_item_train, reid_user_valid, reid_item_valid = \
        load_data(FLAGS.train_data, FLAGS.valid_data, FLAGS.user_review, FLAGS.item_review, FLAGS.user_review_id,
                  FLAGS.item_review_id, FLAGS.stopwords)

    np.random.seed(2017) 

    shuffle_indices = np.random.permutation(np.arange(len(y_train)))

    userid_train = uid_train[shuffle_indices]
    itemid_train = iid_train[shuffle_indices]
    y_train = y_train[shuffle_indices]
    reid_user_train = reid_user_train[shuffle_indices]
    reid_item_train = reid_item_train[shuffle_indices]

    y_train = y_train[:, np.newaxis]
    y_valid = y_valid[:, np.newaxis]

    userid_train = userid_train[:, np.newaxis]
    itemid_train = itemid_train[:, np.newaxis]
    userid_valid = uid_valid[:, np.newaxis]
    itemid_valid = iid_valid[:, np.newaxis]

    batches_train = list(
        zip(userid_train, itemid_train, reid_user_train, reid_item_train, y_train))
    batches_test = list(zip(userid_valid, itemid_valid, reid_user_valid, reid_item_valid, y_valid))
    print 'write begin'
    output = open(os.path.join(TPS_DIR, 'music.train'), 'wb')
    pickle.dump(batches_train, output)
    output = open(os.path.join(TPS_DIR, 'music.test'), 'wb')
    pickle.dump(batches_test, output)

    para = {}
    para['user_num'] = user_num
    para['item_num'] = item_num
    para['review_num_u'] = u_text[0].shape[0]
    para['review_num_i'] = i_text[0].shape[0]
    para['review_len_u'] = u_text[1].shape[1]
    para['review_len_i'] = i_text[1].shape[1]
    para['user_vocab'] = vocabulary_user
    para['item_vocab'] = vocabulary_item
    para['train_length'] = len(y_train)
    para['test_length'] = len(y_valid)
    para['u_text'] = u_text
    para['i_text'] = i_text
    output = open(os.path.join(TPS_DIR, 'music.para'), 'wb')

    # Pickle dictionary using protocol 0.
    pickle.dump(para, output) 