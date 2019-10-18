import torch
import torch.nn as nn
from allennlp.modules.elmo import Elmo, batch_to_ids
from model_han import main_model
import sys

PAD_INDEX = 0
PAD_WORD = '<PAD>'

START_INDEX = 1
START_WORD = '<STR>'

END_INDEX = 2
END_WORD = '<END>'

UNK_INDEX = 3
UNK_WORD = '<UNK>'

# Elmo Parameters
elmo_options = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json" 
elmo_weights = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
elmo = Elmo(elmo_options, elmo_weights, 2, dropout = 0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class KGRAMS(nn.Module):
    def __init__(self, word_embedding_size, vocab_size, out_channels, filter_size, batch_size, review_length, user_id_embedding_idx,
                 item_id_embedding_idx, id_embedding_size, hidden_size, latent_size, num_of_lstm_layers, num_directions, lstm_hidden_dim)    :
        super(KGRAMS, self).__init__()
        self.review_length = review_length
        self.word_embedding_size = word_embedding_size
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=word_embedding_size)
        self.han = main_model(self.word_embedding_size, hidden_size, 2, 0, 0, 0, 0, 0, 0, self.batch_size)
        self.user_net = EntityNet(embedding_id_size=id_embedding_size,
                                  word_embedding_size = word_embedding_size,
                                  out_channels=out_channels,
                                  filter_size=filter_size,
                                  review_length=review_length,
                                  score_size=item_id_embedding_idx,
                                  id_size=user_id_embedding_idx,
                                  hidden_size=hidden_size,
                                  latent_size=latent_size)
        self.item_net = EntityNet(embedding_id_size=id_embedding_size,
                                  word_embedding_size = word_embedding_size,
                                  out_channels=out_channels,
                                  filter_size=filter_size,
                                  review_length=review_length,
                                  score_size=user_id_embedding_idx,
                                  id_size=item_id_embedding_idx,
                                  hidden_size=hidden_size,
                                  latent_size=latent_size)

        #Rating prediction layer
        self.W_1 = nn.Parameter(torch.rand(1, id_embedding_size + latent_size))
        self.b_u = nn.Parameter(torch.rand(batch_size))
        self.b_i = nn.Parameter(torch.rand(batch_size))
        self.mu = nn.Parameter(torch.rand(batch_size))
        #Review generation lstm
        self.num_of_lstm_layers = num_of_lstm_layers
        self.num_directions = num_directions
        self.lstm_hidden_size = lstm_hidden_dim * num_directions * num_of_lstm_layers
        self.linear_layer = nn.Linear(self.lstm_hidden_size, vocab_size)
        self.lstm = nn.LSTM(input_size= (word_embedding_size + id_embedding_size + latent_size), hidden_size=self.lstm_hidden_size, batch_first=True)
        self.linear_c0 = nn.Linear(2 * (id_embedding_size), self.lstm_hidden_size)
        self.linear_h0 = nn.Linear(2 * (id_embedding_size), self.lstm_hidden_size)
        self.lstm_linear_layer = nn.Linear(self.lstm_hidden_size, vocab_size)
        self.tanh = nn.Tanh()

    def initialize_lstm(self, init_values, batch_size):
        c0 = self.tanh(self.linear_c0(init_values)).view(self.num_of_lstm_layers * self.num_directions, batch_size, -1)
        h0 = self.tanh(self.linear_h0(init_values)).view(self.num_of_lstm_layers * self.num_directions, batch_size, -1)
        # 1x5x50
        return (h0, c0)

    def generate_sequence(self, h0, c0, sentiment_feat):
        review_generated = []
        input = torch.LongTensor(self.batch_size, 1).fill_(START_INDEX).to(device)
        input = input.view(input.shape[0], 1, 1)
        # input_embedding = self.word_embeddings(input)
        words_from_ids = convert_batch_indices_to_word(input, self.dataset_object)
        character_ids = batch_to_ids(words_from_ids)
        elmo_embeddings_dict = elmo(character_ids)
        input_embedding = elmo_embeddings_dict['elmo_representations'][0]
        sentiment_feat = sentiment_feat.view(self.batch_size, input_embedding.size(1), -1)#sentiment_feat.repeat(input_embedding.size(1), 1).view(self.batch_size, input_embedding.size(1), -1)
        print("Sentiment Feat", sentiment_feat.shape)
        input_embedding = input_embedding.to(device)
        input_embedding = torch.cat([input_embedding, sentiment_feat], dim=2)
        for i in range(self.review_length):
            hidden_output, (h0, c0) = self.lstm(input_embedding, (h0, c0))
            out_probability = self.lstm_linear_layer(hidden_output)
            word_idx = torch.argmax(out_probability, dim=2)
            word_idx = word_idx.unsqueeze(dim=2)
            # input_embedding = self.word_embeddings(word_idx)#.view(self.batch_size, 1, -1)
            words_from_ids = convert_batch_indices_to_word(word_idx, self.dataset_object)
            character_ids = batch_to_ids(words_from_ids)
            elmo_embeddings_dict = elmo(character_ids)
            input_embedding = elmo_embeddings_dict['elmo_representations'][0]
            input_embedding = input_embedding.to(device)
            print("Size of test input embedding", input_embedding.shape)
            input_embedding = torch.cat([input_embedding, sentiment_feat], dim=2) # [5, 1, 1380]
            print("Word IDx", word_idx.shape)
            word_idx = word_idx.squeeze(dim = 2)
            review_generated.append(word_idx.detach().t())
        review_generated = torch.stack(review_generated, dim=0)
        return review_generated.view(self.batch_size, -1)


    def forward(self, user_ids, user_reviews, user_ids_of_reviews, item_ids, item_reviews, item_ids_of_reviews, target_reviews_x, dataset_object, mode = "train"):
        self.dataset_object = dataset_object
        users_features =  self.user_net(user_ids, user_reviews, item_ids_of_reviews, self.word_embeddings, dataset_object, self.han)
        # 5 x 356
        items_features =  self.item_net(item_ids, item_reviews, user_ids_of_reviews, self.word_embeddings, dataset_object, self.han)
        # 5 x 356
        user_item_features = users_features * items_features  #h_0 = q_u + X_u * p_i + Y_i
        # 5 x 356
        predicted_rating = torch.matmul(self.W_1, user_item_features.t()) + self.b_u + self.b_i + self.mu
        #review generation
        user_id_embeddings = self.user_net.entity_id_embeddings(user_ids)
        item_id_embeddings = self.item_net.entity_id_embeddings(item_ids)
        user_item_id_concatenation = torch.cat([user_id_embeddings, item_id_embeddings], dim=1)
        h0, c0 = self.initialize_lstm(user_item_id_concatenation, self.batch_size)
        if mode is "train":

            # review_embedding = self.word_embeddings(target_reviews_x)
            target_reviews_x = target_reviews_x.view(target_reviews_x.shape[0], target_reviews_x.shape[1], 1)
            words_from_ids = convert_batch_indices_to_word(target_reviews_x, self.dataset_object)
            character_ids = batch_to_ids(words_from_ids)
            elmo_embeddings_dict = elmo(character_ids)
            review_embedding = elmo_embeddings_dict['elmo_representations'][0]
            review_embedding = review_embedding.to(device)
            sentiment_feats = torch.tensor(user_item_features)
            sentiment_feats = sentiment_feats.unsqueeze(dim = 1)
            repeat_size = int(review_embedding.shape[0] / sentiment_feats.shape[0])
            # sentiment_feats =  user_item_features.repeat(review_embedding.size(1), 1).view(self.batch_size, review_embedding.size(1), -1)
            sentiment_feats = sentiment_feats.repeat(repeat_size, 1, 1)#.view(self.batch_size, review_embedding.size(1), -1)
            sentiment_feats = sentiment_feats.to(device)
            lstm_input = torch.cat([review_embedding,sentiment_feats],dim=2)
            lstm_input = lstm_input.squeeze()
            lstm_input = lstm_input.view(self.batch_size,repeat_size,lstm_input.shape[1])
            hidden_ouput, (h0, c0) = self.lstm(lstm_input, (h0, c0))
            out_probability = self.lstm_linear_layer(hidden_ouput)
            return predicted_rating, out_probability
        else:
            seq = self.generate_sequence(h0, c0, user_item_features)
            return predicted_rating, seq


class EntityNet(nn.Module):
    def __init__(self, embedding_id_size, word_embedding_size, out_channels, filter_size, review_length, score_size, id_size, hidden_size, latent_size):
        super(EntityNet, self).__init__()
        self.entity_score_embeddings = nn.Embedding(num_embeddings=score_size,
                                            embedding_dim=embedding_id_size)
        self.entity_id_embeddings = nn.Embedding(num_embeddings=id_size,
                                            embedding_dim=embedding_id_size)
        self.conv2d = nn.Conv2d(in_channels = 1,
                                 out_channels = out_channels,
                                 kernel_size = (filter_size, word_embedding_size),
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
        

    def forward(self, target_id, reviews, review_ids, word_embeddings, dataset_object, han_object):
        batch_size = reviews.size(0)
        num_of_reviews = reviews.size(1)
        review_length = reviews.size(2)
        # ELMO part
        # print(reviews.shape) # 5 x 9 x 82
        review_words = convert_batch_indices_to_word(reviews, dataset_object)
        character_ids = batch_to_ids(review_words)
        elmo_embeddings_dict = elmo(character_ids)
        rev_embeddings = elmo_embeddings_dict['elmo_representations'][0]
        # print(rev_embeddings.shape) # 45 x 82 x 1024
        # input_embeddings = rev_embeddings.view(rev_embeddings.size(0), 1, rev_embeddings.size(1), rev_embeddings.size(2)).to(device)
        # 45 x 1 x 82 x 1024 ; o/p size: batch_size X num_of_reviews X review_length X embedding_size
        # rev_embeddings = word_embeddings(reviews)  
        # input_embeddings = rev_embeddings.view(rev_embeddings.size(0) * rev_embeddings.size(1), 1, rev_embeddings.size(2), rev_embeddings.size(3))
        # x = self.conv2d(input_embeddings)  #o/p size:  num_of_reviews * out_channels * filter_out_size1 * filter_out_size2
        # print("Size of x after conv2d", x.shape) # 45 x 100 X 80 X 925
        # x = self.relu(x)
        # review_feats = self.max_pool(x)
        # print(review_feats.shape) # 45 X 100 X 80 X 1
        # review_feats = review_feats.view(batch_size, num_of_reviews, -1) #4D : [batch_size * num_of_reviews] * out_channels * 1 * 1
        # score_embedding = self.entity_score_embeddings(review_ids).view(batch_size, -1, num_of_reviews) #transposed for matmul
        # print("Size of score embedding", score_embedding.shape) # 5 X 100 X 9
        # review_attentions
        # review_feats = review_feats.view(batch_size, -1, num_of_reviews)
        # print(review_feats.shape) # 5 X 92500 X 9
        # review_attentions = torch.matmul(self.h.view(1, -1), self.relu(torch.matmul(self.W_O, review_feats) + torch.matmul(self.W_u, score_embedding) + self.b1.view(-1,1))) + self.b2
        # review_attentions = self.softmax(review_attentions)  #batch_size X num_of_reviews
        # reviews_importance = torch.matmul(review_attentions, review_feats.view(batch_size, num_of_reviews, -1) )
        # entity_features = self.linear(reviews_importance) #Y_i =W_0 * O_i + b_0        
        # Implementing HAN
        entity_features = han_object(rev_embeddings, num_of_reviews)    
        print("Han output",entity_features.shape) # 5 x 256
        target_id_embedding = self.entity_id_embeddings(target_id).view(batch_size,1, -1) #p_i # [5, 1, 100]
        print(target_id_embedding.shape)
        target_id_embedding = target_id_embedding.squeeze()
        entity_features = torch.cat((target_id_embedding, entity_features),dim=1).view(batch_size, -1)  #batch_size X (embedding_id_size + latent_size)  --- p_i + Y_i
        return entity_features

def convert_batch_indices_to_word(review_indices_batch, dataset_object):

    # Overall List
    batch_element_list = []

    # Iterating through the batch elements
    for element in review_indices_batch:
        # Each element is a 2-D matrix
        for row in range(element.shape[0]):     # total rows = total reviews per user
            temp_review_list = []
            for col in range(element.shape[1]):  # total number of columns = max_rev_length
                temp_review_list.append(dataset_object.idx2word[element[row][col].item()])
            batch_element_list.append(temp_review_list)
    return batch_element_list