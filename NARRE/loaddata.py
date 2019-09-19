'''
Data pre process
@author:
Chong Chen (cstchenc@163.com)
@ created:
25/8/2017
@references:
'''
# Importing the libraries
import os
import json
import pandas as pd
import pickle
import numpy as np

# File paths
TPS_DIR = '/content/NARRE/data/music'
TP_file = os.path.join(TPS_DIR, 'Digital_Music_5.json')

# List for each type
f= open(TP_file)
users_id=[]
items_id=[]
ratings=[]
reviews=[]
np.random.seed(2017)

# For each json line (dict)
for line in f:
    js=json.loads(line) # Gives a Dictionary
    if str(js['reviewerID'])=='unknown':
        print "unknown"
        continue
    if str(js['asin'])=='unknown':
        print "unknown2"
        continue
    reviews.append(js['reviewText']) # Each entry is a string
    users_id.append(str(js['reviewerID'])+',') # Each entry is a string with a comma at the end
    items_id.append(str(js['asin'])+',') # Each entry is a string with a comma at the end
    ratings.append(str(js['overall'])) # String rating
data=pd.DataFrame({'user_id':pd.Series(users_id),
                   'item_id':pd.Series(items_id),
                   'ratings':pd.Series(ratings),
                   'reviews':pd.Series(reviews)})[['user_id','item_id','ratings','reviews']]

def get_count(tp, id): # (dataframe, column_name)
    playcount_groupbyid = tp[[id, 'ratings']].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count # dataframe
usercount, itemcount = get_count(data, 'user_id'), get_count(data, 'item_id') # reviews per user, reviews per item
unique_uid = usercount.index # Index Variable a list containing all the unique user ids
unique_sid = itemcount.index # Index Variable a list containing all the unique item ids
item2id = dict((sid, i) for (i, sid) in enumerate(unique_sid)) # Dict(ItemID: Index)
user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid)) # Dict(UserID: Index)
def numerize(tp): # Argument: Dataframe
    uid = map(lambda x: user2id[x], tp['user_id']) # List of indices pertaining to each user id
    sid = map(lambda x: item2id[x], tp['item_id']) # List of indices pertaining to each item id
    tp['user_id'] = uid # Replacing each column with the user indices
    tp['item_id'] = sid # Replacing each column with the item indices
    return tp

#splitting the data (common processing)
def split_data(data, train_split = 0.7, val_split = 0.15, test_split = 0.15):
        data_size = len(data)
        review_ids = [i for i in range(0, data_size)]
        review_ids = np.random.permutation(review_ids)
        return review_ids

data=numerize(data) # Updated dataframe
tp_rating=data[['user_id','item_id','ratings']] # Extracting particular columns from the dataframe; dataframe

# tp_rating - data frame with only 3 columns
# n_ratings - original size of dataset
# test - np array (20% of the original dataset) with validation indices
# test_idx - same size as original dataset but with validation indices set to true
# tp_1 - dataframe consisting of validation + test (3 columns)
# tp_train - dataframe consisting of train (3 columns)
# data2 - original dataframe with validation 
# data - original dataframe with train

n_ratings = tp_rating.shape[0] # No of rows 
review_ids = split_data(tp_rating) # Permuted indices (reviewIDS)
train_idx = np.zeros(n_ratings, dtype = bool)
train_idx[review_ids[0: int(0.7 * n_ratings)]] = True # Train indices

tp_train = tp_rating[train_idx] # Training data
tp_1 = tp_rating[~train_idx] # Validation + test data
data2 = data[~train_idx]
data = data[train_idx] 

valid_idx = np.zeros(n_ratings, dtype = bool)
# valid_idx[0 : int(0.5 * tp_1.shape[0])] = True
valid_idx[review_ids[int(0.7 * n_ratings) : int(0.7 * n_ratings) + int(0.15 * n_ratings)]] = True
tp_valid = tp_rating[valid_idx]

test_idx = np.zeros(n_ratings, dtype = bool)
test_idx[review_ids[int(0.7 * n_ratings) + int(0.15 * n_ratings)] : ] = True
tp_test = tp_rating[test_idx]

# test = np.random.choice(n_ratings, size=int(0.20 * n_ratings), replace=False) # Random 20% of the train data as validation
# test_idx = np.zeros(n_ratings, dtype=bool) # Initializing all to False
# test_idx[test] = True # Setting the validation indices to True

# tp_1 = tp_rating[test_idx] # Validation (without reviewText) 
# tp_train= tp_rating[~test_idx] # Train (without reviewText)

# data2=data[test_idx] # Validation (with reviewText)
# data=data[~test_idx] # Train (with reviewText)


# n_ratings = tp_1.shape[0] # No of rows in the validation set
# test = np.random.choice(n_ratings, size=int(0.50 * n_ratings), replace=False) # Random 50% of the validation data as test

# test_idx = np.zeros(n_ratings, dtype=bool) # Initializing all to False
# test_idx[test] = True # Setting the test indices to True

#tp_test = tp_1[test_idx] # Test Data
#tp_valid = tp_1[~test_idx] # Validation Data
tp_train.to_csv(os.path.join(TPS_DIR, 'music_train.csv'), index=False,header=None)
tp_valid.to_csv(os.path.join(TPS_DIR, 'music_valid.csv'), index=False,header=None)
tp_test.to_csv(os.path.join(TPS_DIR, 'music_test.csv'), index=False,header=None)

user_reviews={} # Key : list; User Id : [Item Reviews]
item_reviews={} # Key : list; Item Id : [User Reviews]
user_rid={} # Key : list; User Id : [Item IDs]
item_rid={} # Key : list; Item Id : [User IDs]
for i in data.values: # Train data
    if user_reviews.has_key(i[0]):
        user_reviews[i[0]].append(i[3]) # Appending user reviews
        user_rid[i[0]].append(i[1]) # Appending Item ID
    else:
        user_rid[i[0]]=[i[1]] # Maps User ID to a list of Item indices
        user_reviews[i[0]]=[i[3]] # Maps User ID to a list of user reviews
    if item_reviews.has_key(i[1]):
        item_reviews[i[1]].append(i[3]) # Maps Item ID to a list of item reviews
        item_rid[i[1]].append(i[0]) # Maps Item ID to a list of User indices
    else:
        item_reviews[i[1]] = [i[3]] # Appending item reviews
        item_rid[i[1]]=[i[0]] # Appending User ID


for i in data2.values: # Validation
    if user_reviews.has_key(i[0]):
        l=1
    else:
        user_rid[i[0]]=[0] # Set Item ID to zero indicating absence in training set
        user_reviews[i[0]]=['0'] # Set User Review to a dummy string
    if item_reviews.has_key(i[1]):
        l=1
    else:
        item_reviews[i[1]] = [0] # Set User ID to zero indicating absence in training set
        item_rid[i[1]]=['0'] # Set Item Review to a dummy string

pickle.dump(user_reviews, open(os.path.join(TPS_DIR, 'user_review'), 'wb'))
pickle.dump(item_reviews, open(os.path.join(TPS_DIR, 'item_review'), 'wb'))
pickle.dump(user_rid, open(os.path.join(TPS_DIR, 'user_rid'), 'wb'))
pickle.dump(item_rid, open(os.path.join(TPS_DIR, 'item_rid'), 'wb'))

usercount, itemcount = get_count(data, 'user_id'), get_count(data, 'item_id') # Counts per user and item from the training set


print np.sort(np.array(usercount.values)) # sorting the users based on the number of reviews they've written (training)

print np.sort(np.array(itemcount.values)) # sorting the items based on the number of reviews they have received (training)