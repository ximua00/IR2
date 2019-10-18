import torch
import torch.nn as nn
PAD_INDEX = 0
PAD_WORD = '<PAD>'

START_INDEX = 1
START_WORD = '<STR>'

END_INDEX = 2
END_WORD = '<END>'

UNK_INDEX = 3
UNK_WORD = '<UNK>'

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
        self.user_net = EntityNet(embedding_id_size=id_embedding_size,
                                  out_channels=out_channels,
                                  filter_size=filter_size,
                                  review_length=review_length,
                                  score_size=item_id_embedding_idx,
                                  id_size=user_id_embedding_idx,
                                  hidden_size=hidden_size,
                                  latent_size=latent_size)
        self.item_net = EntityNet(embedding_id_size=id_embedding_size,
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
        #Review generaion lstm
        self.num_of_lstm_layers = num_of_lstm_layers
        self.num_directions = num_directions
        self.lstm_hidden_size = lstm_hidden_dim * num_directions * num_of_lstm_layers
        self.linear_layer = nn.Linear(self.lstm_hidden_size, vocab_size)
        self.lstm = nn.LSTM(input_size=word_embedding_size+(id_embedding_size+latent_size), hidden_size=self.lstm_hidden_size, batch_first=True)
        self.linear_c0 = nn.Linear(2 * (id_embedding_size), self.lstm_hidden_size)
        self.linear_h0 = nn.Linear(2 * (id_embedding_size), self.lstm_hidden_size)
        self.lstm_linear_layer = nn.Linear(self.lstm_hidden_size, vocab_size)
        self.tanh = nn.Tanh()

    def initialize_lstm(self, init_values, batch_size):
        c0 = self.tanh(self.linear_c0(init_values)).view(self.num_of_lstm_layers * self.num_directions, batch_size, -1)
        h0 = self.tanh(self.linear_h0(init_values)).view(self.num_of_lstm_layers * self.num_directions, batch_size, -1)
        return (h0, c0)

    def generate_sequence(self, h0, c0, sentiment_feat):
        review_generated = [START_INDEX]
        test_batch_size = 1
        input = torch.LongTensor(test_batch_size, 1).fill_(START_INDEX).to(device)
        input_embedding = self.word_embeddings(input)
        sentiment_feat = sentiment_feat.view(test_batch_size, input_embedding.size(1), -1)#sentiment_feat.repeat(input_embedding.size(1), 1).view(self.batch_size, input_embedding.size(1), -1)
        input_embedding = torch.cat([input_embedding, sentiment_feat], dim=2)
        for i in range(self.review_length):
            hidden_ouput, (h0, c0) = self.lstm(input_embedding, (h0, c0))
            out_probability = self.lstm_linear_layer(hidden_ouput)
            # word_idx = torch.argmax(out_probability.squeeze(), keepdim=True).view(test_batch_size, 1)
            word_weights = out_probability.squeeze().exp()
            word_idx = torch.multinomial(word_weights, 1)[0].view(test_batch_size, 1)
            input_embedding = self.word_embeddings(word_idx)#.view(self.batch_size, 1, -1)
            input_embedding = torch.cat([input_embedding, sentiment_feat], dim=2)
            review_generated.append(word_idx.squeeze().item())
        review_generated.append(END_INDEX)
        return review_generated


    def forward(self, user_ids, user_reviews, user_ids_of_reviews, item_ids, item_reviews, item_ids_of_reviews, target_reviews_x, mode = "train"):
        users_features =  self.user_net(user_ids, user_reviews, item_ids_of_reviews, self.word_embeddings)
        items_features =  self.item_net(item_ids, item_reviews, user_ids_of_reviews, self.word_embeddings)
        user_item_features = users_features * items_features  #h_0 = q_u + X_u * p_i + Y_i
        predicted_rating = torch.matmul(self.W_1, user_item_features.t()) + self.b_u + self.b_i + self.mu
        #review generation
        user_id_embeddings = self.user_net.entity_id_embeddings(user_ids)
        item_id_embeddings = self.item_net.entity_id_embeddings(item_ids)
        user_item_id_concatenation = torch.cat([user_id_embeddings, item_id_embeddings], dim=1)
        if mode is "train":
            h0, c0 = self.initialize_lstm(user_item_id_concatenation, self.batch_size)
            review_embedding = self.word_embeddings(target_reviews_x)
            sentiment_feats =  user_item_features.repeat(review_embedding.size(1), 1).view(self.batch_size, review_embedding.size(1), -1)
            lstm_input = torch.cat([review_embedding,sentiment_feats],dim=2)
            hidden_ouput, (h0, c0) = self.lstm(lstm_input, (h0, c0))
            out_probability = self.lstm_linear_layer(hidden_ouput)
            return predicted_rating, out_probability
        else:
            reviews = []
            for batch in range(self.batch_size):
                h0, c0 = self.initialize_lstm(user_item_id_concatenation[batch,:], 1)
                seq = self.generate_sequence(h0, c0, user_item_features[batch,:])
                reviews.append(seq)
            return predicted_rating, reviews




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
        batch_size = reviews.size(0)
        num_of_reviews = reviews.size(1)
        review_length = reviews.size(2)
        rev_embeddings = word_embeddings(reviews)  # o/p size: batch_size X num_of_reviews X review_length X embedding_size
        input_embeddings = rev_embeddings.view(rev_embeddings.size(0) * rev_embeddings.size(1), 1, rev_embeddings.size(2), rev_embeddings.size(3))
        x = self.conv2d(input_embeddings)  #o/p size:  num_of_reviews * out_channels * filter_out_size1 * filter_out_size2
        x = self.relu(x)
        review_feats = self.max_pool(x).view(batch_size, num_of_reviews, -1) #4D : [batch_size * num_of_reviews] * out_channels * 1 * 1
        score_embedding = self.entity_score_embeddings(review_ids).view(batch_size, -1, num_of_reviews) #transposed for matmul
        # review_attentions
        review_feats = review_feats.view(batch_size, -1, num_of_reviews)
        #Todo: for different embedding sizes
        review_attentions = torch.matmul(self.h.view(1, -1), self.relu(torch.matmul(self.W_O, review_feats) + torch.matmul(self.W_u, score_embedding) + self.b1.view(-1,1))) + self.b2
        review_attentions = self.softmax(review_attentions)  #batch_size X num_of_reviews
        reviews_importance = torch.matmul(review_attentions, review_feats.view(batch_size, num_of_reviews, -1) )
        entity_features = self.linear(reviews_importance) #Y_i =W_0 * O_i + b_0
        target_id_embedding = self.entity_id_embeddings(target_id).view(batch_size,1, -1) #p_i
        entity_features = torch.cat((target_id_embedding, entity_features),dim=2).view(batch_size, -1)  #batch_size X (embedding_id_size + latent_size)  --- p_i + Y_i
        return entity_features