import torch
import torch.nn as nn
import pickle
from KGRAMSData import DataLoader

class KGRAMS(nn.Module):
    def __init__(self, embedding_size, vocab_size, out_channels, filter_size, num_of_user_item_pairs, review_length, user_id_size, item_id_size, embedding_id_size, hidden_size, latent_size)    :
        super(KGRAMS, self).__init__()
        self.review_length = review_length
        self.word_embedding_size = embedding_size
        self.num_of_user_item_pairs = num_of_user_item_pairs
        self.word_embeddings = nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=embedding_size)
        self.user_net = EntityNet(embedding_id_size=embedding_id_size,
                                  out_channels=out_channels,
                                  filter_size=filter_size,
                                  review_length=review_length,
                                  score_size=item_id_size,
                                  id_size=user_id_size,
                                  hidden_size=hidden_size,
                                  latent_size=latent_size)
        self.item_net = EntityNet(embedding_id_size=embedding_id_size,
                                  out_channels=out_channels,
                                  filter_size=filter_size,
                                  review_length=review_length,
                                  score_size=user_id_size,
                                  id_size=item_id_size,
                                  hidden_size=hidden_size,
                                  latent_size=latent_size)

        #Rating prediction layer
        self.W_1 = nn.Parameter(torch.rand(1,embedding_id_size+latent_size))
        self.b_u = nn.Parameter(torch.rand(num_of_user_item_pairs))
        self.b_i = nn.Parameter(torch.rand(num_of_user_item_pairs))
        self.mu = nn.Parameter(torch.rand(num_of_user_item_pairs))
        #Rating component Loss
        self.rating_loss = nn.MSELoss(reduction = 'mean')
        #Review generation part
        #To initialize state of lstm with the item_id+user_id embeddings
        self.linear_c0 = nn.Linear((user_id_size + item_id_size), hidden_size) #hidden size should be number of hidden dimensions of the LSTM state
        self.linear_h0 = nn.Linear((user_id_size + item_id_size), hidden_size)  # hidden size should be number of hidden dimensions of the LSTM state
        self.tanh = nn.Tanh()
        #lstm
        self.num_of_lstm_layers = 1
        self.encoder = nn.LSTM(input_size = embedding_size, hidden_size = hidden_size, batch_first=True)


    def forward(self, user_ids, user_reviews, user_ids_of_reviews, item_ids, item_reviews, item_ids_of_reviews, target_ratings, target_reviews):
        users_features =  self.get_entity_features(user_ids, user_reviews, item_ids_of_reviews, self.word_embeddings, self.user_net)
        items_features =  self.get_entity_features(item_ids, item_reviews, user_ids_of_reviews, self.word_embeddings, self.item_net)
        #element-wise product
        user_item_features = users_features * items_features  #h_0 = q_u + X_u * p_i + Y_i
        #rating prediction
        predicted_rating = torch.matmul(self.W_1, user_item_features.t()) + self.b_u + self.b_i + self.mu
        print("Rating predicted", predicted_rating)
        #calculate loss
        stacked_target_rating = torch.stack(target_ratings, dim=0).view(1, -1)
        rating_pred_loss = self.rating_loss(predicted_rating, stacked_target_rating).squeeze()
        rating_pred_loss.backward() #updates user embedding and item embeddings. Use updated embeddings to initialize lstm
        #LSTM part
        user_embeddings = self.user_net.entity_id_embeddings(torch.stack(user_ids, dim=0).squeeze())   #Updated embeddings
        item_embeddings = self.item_net.entity_id_embeddings(torch.stack(item_ids, dim=0).squeeze())
        # encoded_reviews = self.encode_reviews(user_embeddings, item_embeddings, user_item_features, torch.stack(target_reviews, dim=0))

    def get_entity_features(self, entity_ids, entity_reviews, entity_score_ids, word_embeddings, entity_network):
        entity_features = []
        for index, target_entity_id in enumerate(entity_ids):
            one_entity_revs = entity_reviews[index]  # Tensor of size: #reviews X review_length
            review_entity_score_ids = entity_score_ids[index]  # Tensor of size: #reviews
            i = entity_network(target_entity_id, one_entity_revs, review_entity_score_ids, word_embeddings)
            entity_features.append(i)
        stacked_entity_feats = torch.stack(entity_features, dim=0).view(len(entity_reviews), -1)
        return stacked_entity_feats

    def init_lstm(self, init_values):
        print(init_values.size())
        c0 = self.tanh(self.linear_c0(init_values)).view(self.num_of_lstm_layers * 1, self.num_of_user_item_pairs, -1)
        h0 = self.tanh(self.linear_h0(init_values)).view(self.num_of_lstm_layers * 1, self.num_of_user_item_pairs, -1)
        return (h0, c0)

    def encode_reviews(self, user_embeddings, item_embeddings, sentiment_feats, reviews):
        z1 = torch.cat([user_embeddings, item_embeddings], dim=1)
        (h0, c0) = self.init_lstm(z1)
        #TODO: append every word embedding with sentiment feats
        review_embeddings = self.word_embeddings(reviews).squeeze()
        _, (h0, c0) = self.encoder(review_embeddings, (h0, c0))
        #   concat each word_embedding of the review with the sentiment_feat
        #   lstm output
        #   calculate lstm loss
        #   backpropagate loss - update item and user embedding


class EntityNet(nn.Module):
    def __init__(self, embedding_id_size, out_channels, filter_size, review_length, score_size, id_size, hidden_size, latent_size):
        super(EntityNet, self).__init__()
        self.entity_score_embeddings = nn.Embedding(num_embeddings=score_size,
                                            embedding_dim=embedding_id_size)
        self.entity_id_embeddings = nn.Embedding(num_embeddings=id_size,
                                            embedding_dim=embedding_id_size)
        self.conv2d = nn.Conv2d(in_channels = 1,
                                 out_channels = out_channels,
                                 kernel_size = (filter_size, embedding_id_size),
                                 stride=1)
        self. relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = (review_length - filter_size + 1, 1))

        self.W_O = nn.Parameter(torch.rand(hidden_size, out_channels))
        self.W_u = nn.Parameter(torch.rand(hidden_size, embedding_id_size))
        self.b1 = nn.Parameter(torch.rand(hidden_size))
        self.b2 = nn.Parameter(torch.rand(1))
        self.h = nn.Parameter(torch.rand(hidden_size))

        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(out_channels, latent_size)

    def forward(self, target_id, reviews, review_ids, word_embeddings):
        #Processes all reviews corresponding to one user/item in a batch
        num_of_reviews = reviews.size(0)
        review_length = reviews.size(1)
        rev_embeddings = word_embeddings(reviews)  # o/p size: #reviews X review_length X embedding_size
        input_embeddings = rev_embeddings.view(rev_embeddings.size(0), 1, rev_embeddings.size(1), rev_embeddings.size(2))
        x = self.conv2d(input_embeddings)  #o/p size:  num_of_reviews * out_channels * filter_out_size1 * filter_out_size2
        x = self.relu(x)
        review_feats = self.max_pool(x).view(num_of_reviews, -1).t() #4D : num_of_reviews * out_channels * 1 * 1
        score_embedding = self.entity_score_embeddings(review_ids).view(-1, num_of_reviews)
        review_attentions = torch.mm(self.h.view(1, -1), self.relu(torch.mm(self.W_O, review_feats) + torch.mm(self.W_u, score_embedding) + self.b1.view(-1,1))) + self.b2
        review_attentions = self.softmax(review_attentions)  #1 X num_of_reviews
        reviews_importance = torch.mm(review_attentions, review_feats.t() )
        entity_features = self.linear(reviews_importance) #Y_i =W_0 * O_i + b_0
        target_id_embedding = self.entity_id_embeddings(target_id).view(1, -1) #p_i
        entity_features = torch.cat((target_id_embedding, entity_features),dim=1)  #1 X (embedding_id_size + latent_size)  --- p_i + Y_i
        return entity_features



#Expected datastructure
#User Entity Net needs the following(Similarly for item entity net):
#Note: Inside list structure we have tensors.
#user_ids = [user1_id, user2_id, .....]

# user_reviews = [ [user1_reviews], [user2_reviews],... ]
# user1_reviews = no_of_reviews X review_length

# item_ids_of_reviews = [ [item_ids_of_user1_reviews], [item_ids_of_user2_reviews], ....  ]
# item_ids_of_user1_reviews = no_of_reviews X 1   i,e one item_id for each review of one user


if __name__ == "__main__":
    train_data_path = "../data/Digital_Music_5_train.pkl"
    val_data_path = "../data/Digital_Music_5_val.pkl"
    test_data_path = "../data/Digital_Music_5_test.pkl"
    word2idx_path = "../data/Digital_Music_5_word2idx.pkl"
    idx2word_path = "../data/Digital_Music_5_idx2word.pkl"

    train_data = pickle.load(open(train_data_path, "rb"))
    val_data = pickle.load(open(val_data_path, "rb"))
    test_data = pickle.load(open(test_data_path, "rb"))
    word2idx = pickle.load(open(word2idx_path, "rb"))
    idx2word = pickle.load(open(idx2word_path, "rb"))

    data = {}
    num_of_points = 0
    for rev_id in train_data.keys():
        if num_of_points == 10000:
            break
        data[rev_id] = train_data[rev_id]
        num_of_points += 1
    print(len(data))

    seq_length = 80

    data_loader = DataLoader(data, seq_length, word2idx, idx2word)
    target_user_ids, target_item_ids, target_ratings, target_reviews, user_reviews, item_reviews, review_user_ids, \
    review_item_ids = data_loader.load_data()
    vocab_size = data_loader.vocab_size
    total_num_of_user_item_pairs = len(user_reviews)
    num_of_users = data_loader.user_count +10000
    num_of_items = data_loader.item_count +10000

    print("Vocab size - ", vocab_size, "Num of users - ", num_of_users, "Num of items - ", num_of_items, "Total pairs - ",
          total_num_of_user_item_pairs, "user_reviews size - ", len(user_reviews),"X", user_reviews[0].size())

    model = KGRAMS(embedding_size = 100,
                   vocab_size = vocab_size,
                   out_channels = 100,
                   filter_size = 3,
                   num_of_user_item_pairs = total_num_of_user_item_pairs,
                   review_length = seq_length,
                   user_id_size = num_of_users,
                   item_id_size = num_of_items,
                   embedding_id_size = 100,
                   hidden_size = 50,
                   latent_size=70)

    model(user_ids = target_user_ids,
          user_reviews = user_reviews,
          user_ids_of_reviews = review_user_ids,
          item_ids = target_item_ids,
          item_reviews = item_reviews,
          item_ids_of_reviews = review_item_ids,
          target_ratings = target_ratings,
          target_reviews = target_reviews)

    # print(model)

#
# #dummy inputs
# num_of_revs=9
# item_reviews = [torch.randint(low=0, high=2, size=(num_of_revs,80)), torch.randint(low=0, high=2, size=(num_of_revs,80))]
# user_reviews = [torch.randint(low=0, high=2, size=(num_of_revs,80)), torch.randint(low=0, high=2, size=(num_of_revs,80))]
#
# review_user_ids = [torch.randint(low=0, high=1, size=(num_of_revs,1)), torch.randint(low=0, high=1, size=(num_of_revs,1))]
# review_item_ids = [torch.randint(low=0, high=2, size=(num_of_revs, 1)), torch.randint(low=0, high=2, size=(num_of_revs, 1))]
#
# target_user_id = [torch.randint(low=0, high = 1, size=(1,1)), torch.randint(low=0, high = 1, size=(1,1))]
# target_item_id = [torch.randint(low=0, high=2, size=(1,1)), torch.randint(low=0, high=2, size=(1,1))]
#
# target_ratings = [torch.tensor(2).type(torch.FloatTensor), torch.tensor(4).type(torch.FloatTensor)]
# target_reviews = [torch.randint(low=0, high=2, size=(1,80)), torch.randint(low=0, high=2, size=(1,80))]