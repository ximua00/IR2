import torch
import torch.nn as nn

class KGRAMS(nn.Module):
    def __init__(self, embedding_size, vocab_size, out_channels, filter_size, num_of_user_item_pairs, review_length, user_id_size, item_id_size, embedding_id_size, hidden_size, latent_size):
        super(KGRAMS, self).__init__()
        self.review_length = review_length # Max review length
        self.word_embedding_size = embedding_size # Dimensionality of word embeddings
        self.num_of_user_item_pairs = num_of_user_item_pairs # No of users
        # Word Embedding matrix of shape Vocab_Size X Embedding_size
        self.word_embeddings = nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=embedding_size)
        # An instance of the User Network
        self.user_net = EntityNet(embedding_id_size=embedding_id_size, # Size of user ID embeddings
                                  out_channels=out_channels, # Number of filters
                                  filter_size=filter_size, # Size of filters
                                  review_length=review_length, # Max review length
                                  score_size=item_id_size, # Number of items
                                  id_size=user_id_size, # Number of users
                                  hidden_size=hidden_size, # Size of hidden LSTM units
                                  latent_size=latent_size) # Latent feature size
        # An instance of the Item Network
        self.item_net = EntityNet(embedding_id_size=embedding_id_size, # Size of Item ID embeddings
                                  out_channels=out_channels, # Number of filters
                                  filter_size=filter_size, # Size of filters
                                  review_length=review_length, # Max review length
                                  score_size=user_id_size, # Number of users
                                  id_size=item_id_size, # Numbers of items
                                  hidden_size=hidden_size, # Size of hidden LSTM units
                                  latent_size=latent_size) # Latent feature size
        
        # Parameters of the Rating prediction layer (embedding_id_size + latent_size = n)
        self.W_1 = nn.Parameter(torch.rand(1,embedding_id_size+latent_size)) # 1 X (Embedding_id_size + latent_size)
        self.b_u = nn.Parameter(torch.rand(num_of_user_item_pairs)) # 1 X Number of users
        self.b_i = nn.Parameter(torch.rand(num_of_user_item_pairs)) # 1 X Number of users
        self.mu = nn.Parameter(torch.rand(num_of_user_item_pairs)) # 1 X numbers of users

        # Rating component Loss - Mean Square Error loss
        self.rating_loss = nn.MSELoss(reduction = 'mean')

        # # Review generation part
        # #To initialize state of lstm with the item_id+user_id embeddings
        # self.linear_c0 = nn.Linear((2 * (embedding_id_size + latent_size)), hidden_size) #hidden size should be number of hidden dimensions of the LSTM state
        # self.linear_h0 = nn.Linear((2 * (embedding_id_size + latent_size)), hidden_size)  # hidden size should be number of hidden dimensions of the LSTM state
        # self.tanh = nn.Tanh()
        # #lstm
        # self.num_of_lstm_layers = 1
        # self.encoder = nn.LSTM(input_size = embedding_size, hidden_size = hidden_size, batch_first=True)


    def forward(self, user_ids, user_reviews, user_ids_of_reviews, item_ids, item_reviews, item_ids_of_reviews, target_ratings, target_reviews):
        # Get the overall user latent representation
        users_features =  self.get_entity_features(user_ids, user_reviews, item_ids_of_reviews, self.word_embeddings, self.user_net)
        # Get the overall item latent representation
        items_features =  self.get_entity_features(item_ids, item_reviews, user_ids_of_reviews, self.word_embeddings, self.item_net)
        # Element-wise product
        user_item_features = users_features * items_features
        # print("user features size: ", users_features.size(), "Item features size: ", items_features.size())
        # Rating prediction
        predicted_rating = torch.matmul(self.W_1, user_item_features.t()) + self.b_u + self.b_i + self.mu
        # Sentiment features????
        # Calculate loss
        stacked_target_rating = torch.stack(target_ratings, dim=0).view(1, -1)
        rating_pred_loss = self.rating_loss(predicted_rating, stacked_target_rating).squeeze()
        rating_pred_loss.backward()
        #LSTM part
        # encoded_reviews = self.encode_reviews(users_features, items_features, torch.stack(target_reviews, dim=0))

    def get_entity_features(self, entity_ids, entity_reviews, entity_score_ids, word_embeddings, entity_network):
        entity_features = []
        for index, target_entity_id in enumerate(entity_ids):
            one_entity_revs = entity_reviews[index]  # Tensor of size: # reviews X max_review_length
            review_entity_score_ids = entity_score_ids[index]  # Tensor of size: Number of items for which the user has written a review for
            i = entity_network(target_entity_id, one_entity_revs, review_entity_score_ids, word_embeddings)
            entity_features.append(i)
        # Stacking the learnt features
        stacked_entity_feats = torch.stack(entity_features, dim=0).view(len(entity_reviews), -1)
        return stacked_entity_feats

    def init_lstm(self, init_values):
        c0 = self.tanh(self.linear_c0(init_values)).view(self.num_of_lstm_layers * 1, self.num_of_user_item_pairs, -1)
        h0 = self.tanh(self.linear_h0(init_values)).view(self.num_of_lstm_layers * 1, self.num_of_user_item_pairs, -1)
        return (h0, c0)

    def encode_reviews(self, users_features, items_features, reviews):
        z1 = torch.cat([users_features, items_features], dim=1)
        (h0, c0) = self.init_lstm(z1)
        review_embeddings = self.word_embeddings(reviews).squeeze()
        _, (h0, c0) = self.encoder(review_embeddings, (h0, c0))
        #   concat each word_embedding of the review with the sentiment_feat
        #   lstm output
        #   calculate lstm loss
        #   backpropagate loss - update item and user embedding


class EntityNet(nn.Module):
    def __init__(self, embedding_id_size, out_channels, filter_size, review_length, score_size, id_size, hidden_size, latent_size):
        super(EntityNet, self).__init__()
        # Embedding matrix for storing the ID representations of items (in case of User Net) or users (in case of Item Net)
        self.entity_score_embeddings = nn.Embedding(num_embeddings=score_size,
                                            embedding_dim=embedding_id_size)
        # Embedding matrix for storing the ID representations of vice versa
        self.entity_id_embeddings = nn.Embedding(num_embeddings=id_size,
                                            embedding_dim=embedding_id_size)
        # Convolutional Layer
        self.conv2d = nn.Conv2d(in_channels = 1,
                                 out_channels = out_channels,
                                 kernel_size = (filter_size, embedding_id_size),
                                 stride=1)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = (review_length - filter_size + 1, 1))

        self.W_O = nn.Parameter(torch.rand(hidden_size, out_channels))
        self.W_u = nn.Parameter(torch.rand(hidden_size, embedding_id_size))
        self.b1 = nn.Parameter(torch.rand(hidden_size))
        self.b2 = nn.Parameter(torch.rand(1))
        self.h = nn.Parameter(torch.rand(hidden_size))

        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(out_channels, latent_size)

    def forward(self, target_id, reviews, review_ids, word_embeddings):
        # Processes all reviews corresponding to one user/item in a batch
        num_of_reviews = reviews.size(0) # reviews is a tensor of size : num_reviews X max_review_length
        rev_embeddings = word_embeddings(reviews)   # o/p size: # num_reviews X max_review_length X embedding_size
        # num_reviews X 1 X max_review_length X embedding_size
        input_embeddings = rev_embeddings.view(rev_embeddings.size(0), 1, rev_embeddings.size(1), rev_embeddings.size(2))
        
        # num_reviews x num_filters X filter_out_size_1 X filter_out_size_2
        x = self.conv2d(input_embeddings)
        x = self.relu(x)

        # num_reviews X num_filters X 1 X 1
        # After squeezing and transposing -> num_filters X num_reviews
        review_feats = self.max_pool(x).squeeze().t()

        # entity_score_embeddings returns num_users / num_items X id_embedding_size
        # Assuming that each user / item has max_num_reviews
        # score_embedding : embedding_size X max_num_reviews
        score_embedding = self.entity_score_embeddings(review_ids).view(-1, num_of_reviews) # ? 
        
        # pre-softmax attention scores of shape 1 X max_num_reviews
        review_attentions = torch.mm(self.h.view(1, -1), self.relu(torch.mm(self.W_O, review_feats) + torch.mm(self.W_u, score_embedding) + self.b1.view(-1,1))) + self.b2
        
        # Softmax over the pre-attention scores of shape 1 x max_num_reviews
        review_attentions = self.softmax(review_attentions)

        # 1 X max_num_reviews * max_num_reviews * num_filters = 1 X num_filters
        reviews_importance = torch.mm(review_attentions, review_feats.t())

        # Using the attention scores to compute overall entity representation
        entity_features = self.linear(reviews_importance) 

        # Get the embedding for the target entity of shape 1 X id_embedding_size
        target_id_embedding = self.entity_id_embeddings(target_id).view(1, -1)

        # Concatenating the overall entity representation + entity id embedding of shape 1 X (latent_size + id_embedding_size)
        entity_features = torch.cat((target_id_embedding, entity_features),dim=1)
        
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
    #dummy inputs

    # Number of reviews
    number_of_reviews = 9

    # Maximum review length
    max_review_length = 80

    # List of Tensors : Each tensor (corresponding to an item) is of shape number_of_reviews X max_review_length
    # Each row is a review consisting of word indices (0, 1, 2)
    item_reviews = [torch.randint(low=0, high=3, size=(number_of_reviews, max_review_length)), torch.randint(low=0, high=3, size=(number_of_reviews, max_review_length))]

    # List of Tensors : Each tensor (corresponding to a user) is of shape number_of_reviews X max_review_length 
    # Each row is a review consisting of word indices (0, 1, 2)
    user_reviews = [torch.randint(low=0, high=3, size=(number_of_reviews, max_review_length)), torch.randint(low=0, high=3, size=(number_of_reviews, max_review_length))]

    # List of Tensors: Each Tensor corresponding to an item containing the IDs of the users who have written a review on it 
    # [[user ids for item 1], [user ids for item 2], ...]
    review_user_ids = [torch.randint(low=0, high=2, size=(number_of_reviews,1)), torch.randint(low=0, high=2, size=(number_of_reviews,1))]
    
    # List of Tensors: Each Tensor corresponding to a user containing the IDs of the items for which they have written a review 
    # [[item ids for user 1], [item ids for user 2]]
    review_item_ids = [torch.randint(low=0, high=3, size=(number_of_reviews, 1)), torch.randint(low=0, high=3, size=(number_of_reviews, 1))]

    # User and Item ID pairs for which you want to predict ratings
    target_user_id = [torch.randint(low=0, high=2, size=(1,1)), torch.randint(low=0, high=2, size=(1,1))]
    target_item_id = [torch.randint(low=0, high=3, size=(1,1)), torch.randint(low=0, high=3, size=(1,1))]

    # Ground truth ratings and reviews
    target_ratings = [torch.tensor(2).type(torch.FloatTensor), torch.tensor(4).type(torch.FloatTensor)]
    target_reviews = [torch.randint(low=0, high=3, size=(1,80)), torch.randint(low=0, high=3, size=(1,80))]

    # Total number of users
    total_num_of_users = len(user_reviews)

    # Instantiating an object of the KGRAMS class
    # Embedding size = 100
    # Vocab size = 3
    # out_channels = 100 (Number of filters)
    # filter size = 3
    # User Item pairs = total number of users
    # Review length = max_review_length = 80
    # User ID size = Number of users = 2
    # Item ID size = Number of items = 3
    # Size of Item and User embeddings = 100
    # Hidden size for LSTM = 50
    # Latent size = 70
    model = KGRAMS(embedding_size = 100, vocab_size = 3, out_channels = 100, filter_size = 3, num_of_user_item_pairs= total_num_of_users, review_length = max_review_length, user_id_size = 2, item_id_size = 3, embedding_id_size = 100, hidden_size = 50, latent_size=70)

    model(user_ids = target_user_id, # User IDs for which you want to predict ratings
          user_reviews = user_reviews, # User reviews from the training set
          user_ids_of_reviews = review_user_ids, # User IDs who have written a review for each item
          item_ids = target_item_id, # Item IDs for which you want to predict ratings
          item_reviews = item_reviews, # Item reviews from the training set
          item_ids_of_reviews = review_item_ids, # Item IDs for which a user has written a review
          target_ratings = target_ratings, # Ground truth ratings
          target_reviews = target_reviews) # Ground truth reviews

    # print(model)