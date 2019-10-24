import torch
import torch.nn as nn
# from allennlp.modules.elmo import Elmo, batch_to_ids
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
# elmo_options = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json" 
# elmo_weights = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
# elmo = Elmo(elmo_options, elmo_weights, 2, dropout = 0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class KGRAMS(nn.Module):
    def __init__(self, word_embedding_size, vocab_size, out_channels, filter_size, batch_size, review_length, user_id_embedding_idx,
                 item_id_embedding_idx, id_embedding_size, hidden_size, latent_size, num_of_lstm_layers, num_directions, lstm_hidden_dim, glove_embedding_matrix)    :
        super(KGRAMS, self).__init__()
        self.review_length = review_length
        self.word_embedding_size = word_embedding_size
        self.batch_size = batch_size

        #### VARIABLE FOR GLOVE EMBEDDINGS ####
        self.glove_embedding_matrix = glove_embedding_matrix
        ####################################################

        self.word_embeddings = nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=word_embedding_size)
        self.user_net = EntityNet(embedding_id_size=id_embedding_size,
                                  word_embedding_size = word_embedding_size,
                                  out_channels=out_channels,
                                  filter_size=filter_size,
                                  review_length=review_length,
                                  score_size=item_id_embedding_idx,
                                  id_size=user_id_embedding_idx,
                                  hidden_size=hidden_size,
                                  latent_size=latent_size,
                                  glove_embedding_matrix = glove_embedding_matrix)
        self.item_net = EntityNet(embedding_id_size=id_embedding_size,
                                  word_embedding_size = word_embedding_size,
                                  out_channels=out_channels,
                                  filter_size=filter_size,
                                  review_length=review_length,
                                  score_size=user_id_embedding_idx,
                                  id_size=item_id_embedding_idx,
                                  hidden_size=hidden_size,
                                  latent_size=latent_size,
                                  glove_embedding_matrix = glove_embedding_matrix)

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
        review_generated = [START_INDEX]
        test_batch_size = 1
        input = torch.LongTensor(test_batch_size, 1).fill_(START_INDEX).to(device)
        # print("Input : {}".format(input.shape)) #[1, 1]

        input_embedding = self.glove_embedding_matrix(input) #[1, 1, 100]
        # print("Input Embedding : {}".format(input_embedding.shape))

        # print("Sentiment Features : {}".format(sentiment_feat.shape)) #[356]

        sentiment_feat = sentiment_feat.view(test_batch_size, input_embedding.size(1), -1)#sentiment_feat.repeat(input_embedding.size(1), 1).view(self.batch_size, input_embedding.size(1), -1)
        # print("Sentiment Features Reshaped: {}".format(sentiment_feat.shape)) #[1, 1, 356]

        input_embedding = input_embedding.to(device)
        input_embedding = torch.cat([input_embedding, sentiment_feat], dim=2)
        # print("Input Embedding Reshaped: {}".format(input_embedding.shape)) #[1, 1, 456]

        for i in range(self.review_length):
            hidden_ouput, (h0, c0) = self.lstm(input_embedding, (h0, c0))
            out_probability = self.lstm_linear_layer(hidden_ouput)
            # word_idx = torch.argmax(out_probability, dim=2)
            word_weights = out_probability.squeeze().exp()
            word_idx = torch.multinomial(word_weights, 1)[0].view(test_batch_size, 1)
            # print("Word IDX : {}".format(word_idx.shape)) #[1, 1]

            input_embedding = self.glove_embedding_matrix(word_idx)
            # print("Input Embedding : {}".format(input_embedding.shape)) #[1, 1, 100]

            input_embedding = input_embedding.to(device)
            input_embedding = torch.cat([input_embedding, sentiment_feat], dim=2) #[1, 1, 456]

            # review_generated.append(word_idx.detach().t())
            review_generated.append(word_idx.squeeze().item())
        # review_generated = torch.stack(review_generated, dim=0)
        review_generated.append(END_INDEX)
        # return review_generated.view(self.batch_size, -1)
        return review_generated


    def forward(self, user_ids, user_reviews, user_ids_of_reviews, item_ids, item_reviews, item_ids_of_reviews, target_reviews_x, dataset_object, mode = "train"):
        
        self.dataset_object = dataset_object
        users_features =  self.user_net(user_ids, user_reviews, item_ids_of_reviews, self.word_embeddings, dataset_object)
        items_features =  self.item_net(item_ids, item_reviews, user_ids_of_reviews, self.word_embeddings, dataset_object)
        user_item_features = users_features * items_features  #h_0 = q_u + X_u * p_i + Y_i
        # print("User Item Features : {}".format(user_item_features.shape)) # [5, 356]

        predicted_rating = torch.matmul(self.W_1, user_item_features.t()) + self.b_u + self.b_i + self.mu
        
        #review generation
        user_id_embeddings = self.user_net.entity_id_embeddings(user_ids)
        item_id_embeddings = self.item_net.entity_id_embeddings(item_ids)
        user_item_id_concatenation = torch.cat([user_id_embeddings, item_id_embeddings], dim=1)
        # print("User Item ID Concatenation : {}".format(user_item_id_concatenation.shape)) # [5, 200]
        
        if mode is "train":
            h0, c0 = self.initialize_lstm(user_item_id_concatenation, self.batch_size) #[1, 5, 128]
            # print("H0 : {}".format(h0.shape))
            # print("C0 : {}".format(c0.shape))

            # print("Target Reviews : {}".format(target_reviews_x.shape)) #[5, 81]
            
            review_embedding = self.glove_embedding_matrix(target_reviews_x) #[5, 81, 100]
            # print("Review Embeddings : {}".format(review_embedding.shape))
            review_embedding = review_embedding.to(device)

            sentiment_feats = torch.tensor(user_item_features)
            sentiment_feats = sentiment_feats.unsqueeze(dim = 1) #[5, 1, 356]
            # print("Sentiment Features : {}".format(sentiment_feats.shape))

            repeat_size = int(review_embedding.shape[1] / sentiment_feats.shape[1])
            # print("Repeat Size : {}".format(repeat_size))

            sentiment_feats = sentiment_feats.repeat(1, repeat_size, 1)#.view(self.batch_size, review_embedding.size(1), -1)
            sentiment_feats = sentiment_feats.to(device)
            # print("Sentiment Features Reshaped : {}".format(sentiment_feats.shape)) #[5, 81, 356]
            

            lstm_input = torch.cat([review_embedding,sentiment_feats],dim=2)
            lstm_input = lstm_input.squeeze()
            # print("LSTM Input : {}".format(lstm_input.shape)) # [5, 81, 456]

            # lstm_input = lstm_input.view(self.batch_size, repeat_size, lstm_input.shape[1])
            # print("LSTM Input Reshaped : {}".format(lstm_input.shape))

            hidden_ouput, (h0, c0) = self.lstm(lstm_input, (h0, c0))
            out_probability = self.lstm_linear_layer(hidden_ouput)
            # print("Out Probability : {}".format(out_probability.shape)) #[5, 81, 5040]

            return predicted_rating, out_probability
        else:
            reviews = []
            for batch in range(self.batch_size):
                h0, c0 = self.initialize_lstm(user_item_id_concatenation[batch,:], 1)
                seq = self.generate_sequence(h0, c0, user_item_features[batch,:])
                reviews.append(seq)
            return predicted_rating, reviews
            # seq = self.generate_sequence(h0, c0, user_item_features)
            # return predicted_rating, seq


class EntityNet(nn.Module):
    def __init__(self, embedding_id_size, word_embedding_size, out_channels, filter_size, review_length, score_size, id_size, hidden_size, latent_size, glove_embedding_matrix):
        super(EntityNet, self).__init__()
        self.entity_score_embeddings = nn.Embedding(num_embeddings=score_size,
                                            embedding_dim=embedding_id_size)
        self.entity_id_embeddings = nn.Embedding(num_embeddings=id_size,
                                            embedding_dim=embedding_id_size)
        
        #### VARIABLE FOR GLOVE EMBEDDINGS ####
        self.glove_embedding_matrix = glove_embedding_matrix
        ####################################################

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
        

    def forward(self, target_id, reviews, review_ids, word_embeddings, dataset_object):
        batch_size = reviews.size(0)
        num_of_reviews = reviews.size(1)
        review_length = reviews.size(2)
        # print("Reviews Shape : {}".format(reviews.shape)) # [5, 8, 82]

        #### Converting to Glove Embeddings ####
        rev_embeddings = self.glove_embedding_matrix(reviews)
        # print("Review Embeddings : {}".format(rev_embeddings.shape)) # [5, 8, 82, 100]
        #############################################################
         
        input_embeddings = rev_embeddings.view(rev_embeddings.size(0) * rev_embeddings.size(1), 1, rev_embeddings.size(2), rev_embeddings.size(3)) # [40, 1, 82, 100]
        # print("Input Embeddings : {}".format(input_embeddings.shape))

        x = self.conv2d(input_embeddings)  #o/p size:  num_of_reviews * out_channels * filter_out_size1 * filter_out_size2
        # print("Size of x after conv2d", x.shape) # 40 x 100 X 80 X 1

        x = self.relu(x)
        review_feats = self.max_pool(x)
        # print("Review Feats : {}".format(review_feats.shape)) # 40 X 100 X 1 X 1 #4D : [batch_size * num_of_reviews] * out_channels * 1 * 1

        score_embedding = self.entity_score_embeddings(review_ids).view(batch_size, -1, num_of_reviews) #transposed for matmul
        # print("Size of score embedding : {}".format(score_embedding.shape)) # 5 X 100 X 8

        # review_attentions
        review_feats = review_feats.view(batch_size, -1, num_of_reviews)
        # print("Review Feats after reshaping : {}".format(review_feats.shape)) # 5 X 100 X 8

        review_attentions = torch.matmul(self.h.view(1, -1), self.relu(torch.matmul(self.W_O, review_feats) + torch.matmul(self.W_u, score_embedding) + self.b1.view(-1,1))) + self.b2
        review_attentions = self.softmax(review_attentions)  # batch_size X num_of_reviews
        # print("Review Attentions : {}".format(review_attentions.shape)) # [5, 1, 8]

        reviews_importance = torch.matmul(review_attentions, review_feats.view(batch_size, num_of_reviews, -1) )
        entity_features = self.linear(reviews_importance) #Y_i =W_0 * O_i + b_0
        # print("Only Entity Features : {}".format(entity_features.shape)) # [5, 1, 256]

        target_id_embedding = self.entity_id_embeddings(target_id).view(batch_size,1, -1) #p_i
        entity_features = torch.cat((target_id_embedding, entity_features),dim=2).view(batch_size, -1)  #batch_size X (embedding_id_size + latent_size)  --- p_i + Y_i
        # print("Entity Features Combined: {}".format(entity_features.shape)) # [5, 356]

        return entity_features