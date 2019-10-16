# Importing Libraries
import torch
import torch.nn as nn
import config
from pyrouge import Rouge155

def obtain_pyrouge_scores():

    # Creating an instance of the rouge object 
    r = Rouge155()

    # Setting the directory paths
    r.system_dir = 'decoded'
    r.model_dir = 'reference'

    # Defining the file patterns in the directories
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_filename_pattern = '#ID#_reference.txt'

    # Obtain the metrics
    output = r.convert_and_evaluate()
    output_dict = r.output_to_dict(output)
    
    # Returning the output dictionary
    return output_dict

def view_generated_reviews(raw_probabilities):

    # Soft-maxing the raw probabilities
    # print(raw_probabilities.shape)
    numerator = torch.exp(raw_probabilities)
    denominator = torch.sum(torch.exp(raw_probabilities), dim = 1)
    raw_probabilities = torch.div(numerator.t(), denominator).t()
    # raw_probabilities = torch.exp(raw_probabilities) / torch.sum(torch.exp(raw_probabilities), dim = 1)

    # Taking the max index
    max_indices = torch.argmax(raw_probabilities, dim = 1)

    # List of generated sentences
    generated_sentences = []

    # Generating the sentences
    for user_item_pair_index in range(max_indices.shape[0]):
        generated_sentence = []
        for word_index in max_indices[user_item_pair_index]:
            generated_sentence.append(id_to_word(word_index))
        print(' '.join(generated_sentence))
        # generated_sentences.append(generated_sentence)

    return max_indices
    
class MRG(nn.Module):

    # Init function
    def __init__(self, word_embedding_size, id_embedding_size, hidden_dim, latent_size, vocab_size, num_of_lstm_layers = 1, num_directions = 1):
        super(MRG, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, word_embedding_size)
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_of_lstm_layers = num_of_lstm_layers
        self.num_directions = num_directions
        self.id_embedding_size = id_embedding_size
        self.lstm_hidden_size = hidden_dim * num_directions * num_of_lstm_layers
        self.linear_layer = nn.Linear(self.lstm_hidden_size, vocab_size)
        self.lstm = nn.LSTM(input_size = word_embedding_size, hidden_size = self.lstm_hidden_size, batch_first = True)
        # MRG linear layers for h0 and c0 + activation function
        self.linear_c0 = nn.Linear(2 * (id_embedding_size + latent_size), self.lstm_hidden_size)
        self.linear_h0 = nn.Linear(2 * (id_embedding_size + latent_size), self.lstm_hidden_size)
        self.tanh = nn.Tanh()

    # Function to initialize the hidden and cell states
    def initialize_lstm(self, init_values, num_of_user_item_pairs):
        c0 = self.tanh(self.linear_c0(init_values)).view(self.num_of_lstm_layers * self.num_directions, num_of_user_item_pairs, -1)
        h0 = self.tanh(self.linear_h0(init_values)).view(self.num_of_lstm_layers * self.num_directions, num_of_user_item_pairs, -1)
        return (h0, c0)
        
    # Forward function
    def forward(self, user_features, item_features, input_reviews, num_of_user_item_pairs):

        # Obtaining user_id_embeddings and the item_id_embeddings
        user_id_embeddings = user_features[0 : self.id_embedding_size]
        item_id_embeddings = item_features[0 : self.id_embedding_size]

        # Creating a concatenation of the user and ID embeddings
        user_item_id_concatenation = torch.cat([user_id_embeddings, item_id_embeddings], dim = 1)
        
        # Initializing the hidden and cell states
        h0, c0 = self.initialize_lstm(user_item_id_concatenation, num_of_user_item_pairs)

        # Converting to tensor
        input_reviews = torch.stack(input_reviews, dim = 0)
        input_reviews = input_reviews.type(torch.LongTensor)
        input_reviews = input_reviews.squeeze()
        input_reviews = self.embedding_layer(input_reviews)

        # Passing the input reviews through the LSTM
        all_hidden_state_outputs, (h0, c0) = self.lstm(input_reviews, (h0, c0))
        
        # Obtaining pre-softmax probabilities over the vocabulary
        raw_probabilities = self.linear_layer(all_hidden_state_outputs) 
        
        return raw_probabilities
        

class NARRE(nn.Module):
    def __init__(self, embedding_size, vocab_size, out_channels, filter_size, review_length, user_id_size, item_id_size, id_embedding_size, hidden_size, latent_size):
        super(NARRE, self).__init__()
        self.review_length = review_length # Max review length
        self.word_embedding_size = embedding_size # Dimensionality of word embeddings
        # Word Embedding matrix of shape Vocab_Size X Embedding_size
        self.word_embeddings = nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=embedding_size)
        # Creating a variable for storing the embedding id size
        self.id_embedding_size = id_embedding_size
        self.latent_size = latent_size

        # An instance of the User Network
        self.user_net = EntityNet(embedding_id_size=id_embedding_size, # Size of user ID embeddings
                                  out_channels=out_channels, # Number of filters
                                  filter_size=filter_size, # Size of filters
                                  review_length=review_length, # Max review length
                                  score_size=item_id_size, # Number of items
                                  id_size=user_id_size, # Number of users
                                  hidden_size=hidden_size, # Size of hidden LSTM units
                                  latent_size=latent_size) # Latent feature size
        
        # An instance of the Item Network
        self.item_net = EntityNet(embedding_id_size=id_embedding_size, # Size of Item ID embeddings
                                  out_channels=out_channels, # Number of filters
                                  filter_size=filter_size, # Size of filters
                                  review_length=review_length, # Max review length
                                  score_size=user_id_size, # Number of users
                                  id_size=item_id_size, # Numbers of items
                                  hidden_size=hidden_size, # Size of hidden LSTM units
                                  latent_size=latent_size) # Latent feature size

    def initialize_rating_prediction_layer(self, num_of_user_item_pairs):
        
        # Parameters of the Rating prediction layer (embedding_id_size + latent_size = n)
        self.W_1 = nn.Parameter(torch.rand(1, self.id_embedding_size + self.latent_size)) # 1 X (Embedding_id_size + latent_size)
        self.b_u = nn.Parameter(torch.rand(num_of_user_item_pairs)) # 1 X Number of user item pairs
        self.b_i = nn.Parameter(torch.rand(num_of_user_item_pairs)) # 1 X Number of user item pairs
        self.mu = nn.Parameter(torch.rand(num_of_user_item_pairs)) # 1 X numbers of user item pairs
        
    def forward(self, user_ids, user_reviews, user_ids_of_reviews, item_ids, item_reviews, item_ids_of_reviews, target_ratings, target_reviews, num_of_user_item_pairs):
        
        # Initializing the parameters of the rating prediction layer
        self.initialize_rating_prediction_layer(num_of_user_item_pairs)

        # Get the overall user latent representation 
        users_features =  self.get_entity_features(user_ids, user_reviews, item_ids_of_reviews, self.word_embeddings, self.user_net)
        # Get the overall item latent representation
        # print(len(item_ids))
        items_features =  self.get_entity_features(item_ids, item_reviews, user_ids_of_reviews, self.word_embeddings, self.item_net)
        # Element-wise product
        user_item_features = users_features * items_features #[2, 170]
        
        predicted_rating = torch.matmul(self.W_1, user_item_features.t()) + self.b_u + self.b_i + self.mu
       
        return predicted_rating, users_features, items_features, user_item_features

    def get_entity_features(self, entity_ids, entity_reviews, entity_score_ids, word_embeddings, entity_network):
        entity_features = []
        for index, target_entity_id in enumerate(entity_ids):
            one_entity_revs = entity_reviews[index]  # Tensor of size: # reviews X max_review_length
            review_entity_score_ids = entity_score_ids[index]  # Tensor of size: Number of items for which the user has written a review for
            i = entity_network(target_entity_id, one_entity_revs, review_entity_score_ids, word_embeddings)
            entity_features.append(i)
        # Stacking the learnt features
        # print(entity_features.shape)
        stacked_entity_feats = torch.stack(entity_features, dim = 0) # [2, 1, 170]
        stacked_entity_feats = stacked_entity_feats.view(len(entity_reviews), -1) # [2, 170]
        # print(stacked_entity_feats.shape)
        # stacked_entity_feats = torch.stack(entity_features, dim=0).view(len(entity_reviews), -1)
        return stacked_entity_feats

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
        reviews = reviews.type(torch.LongTensor)
        rev_embeddings = word_embeddings(torch.LongTensor(reviews))   # o/p size: # num_reviews X max_review_length X embedding_size
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
        review_ids = review_ids.type(torch.LongTensor)
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
        target_id = target_id.type(torch.LongTensor)
        target_id_embedding = self.entity_id_embeddings(target_id).view(1, -1)

        # Concatenating the overall entity representation + entity id embedding of shape 1 X (latent_size + id_embedding_size)
        entity_features = torch.cat((target_id_embedding, entity_features),dim=1)
        # print(entity_features.shape)
        
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

    # List of Tensors : Each tensor (corresponding to an item) is of shape number_of_reviews X max_review_length
    # Each tensor corresponds to a datapoint which contains all the reviews on that item which is a part of that datapoint
    # Each row is a review consisting of word indices (0, 1, 2)
    item_reviews = [torch.randint(low=0, high=3, size=(config.max_reviews, config.max_review_length)), torch.randint(low=0, high=3, size=(config.max_reviews, config.max_review_length))]

    # List of Tensors : Each tensor (corresponding to a user) is of shape number_of_reviews X max_review_length 
    # Each row is a review consisting of word indices (0, 1, 2)
    # Each tensor corresponds to a datapoint which contains all the reviews of that user who is a part of that datapoint
    user_reviews = [torch.randint(low=0, high=3, size=(config.max_reviews, config.max_review_length)), torch.randint(low=0, high=3, size=(config.max_reviews, config.max_review_length))]

    # List of Tensors: Each Tensor corresponding to an item containing the IDs of the users who have written a review on it 
    # Each tensor corresponds to an element of item reviews which contains all the user IDS that wrote each review in that element 
    review_user_ids = [torch.randint(low=0, high=2, size=(config.max_reviews, 1)), torch.randint(low=0, high=2, size=(config.max_reviews, 1))]
    
    # List of Tensors: Each Tensor corresponding to a user containing the IDs of the items for which they have written a review 
    # Each tensor corresponds to an element of user reviws which contains all the item IDS for which each review was written in that element 
    review_item_ids = [torch.randint(low=0, high=3, size=(config.max_reviews, 1)), torch.randint(low=0, high=3, size=(config.max_reviews, 1))]

    # User and Item ID pairs for which you want to predict ratings
    target_user_id = [torch.randint(low=0, high=2, size=(1,1)), torch.randint(low=0, high=2, size=(1,1))]
    target_item_id = [torch.randint(low=0, high=3, size=(1,1)), torch.randint(low=0, high=3, size=(1,1))]

    # Ground truth ratings and reviews
    target_ratings = [torch.tensor(2).type(torch.FloatTensor), torch.tensor(4).type(torch.FloatTensor)]
    target_reviews = [torch.randint(low=0, high=3, size=(1,80)), torch.randint(low=0, high=3, size=(1,80))]

    # Total number of users
    num_of_user_item_pairs = len(user_reviews)

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
    narre = NARRE(embedding_size = config.word_embedding_size, vocab_size = config.vocab_size, out_channels = config.num_filters, filter_size = config.filter_size, review_length = config.max_review_length, user_id_size = config.num_users, item_id_size = config.num_items, id_embedding_size = config.id_embedding_size, hidden_size = config.hidden_size, latent_size = config.latent_size)
    mrg = MRG(word_embedding_size = config.word_embedding_size, id_embedding_size = config.id_embedding_size, hidden_dim = config.hidden_size, latent_size = config.latent_size, vocab_size = config.vocab_size)
    predicted_ratings, user_features, item_features, user_item_features = narre(user_ids = target_user_id, # User IDs for which you want to predict ratings
          user_reviews = user_reviews, # User reviews from the training set
          user_ids_of_reviews = review_user_ids, # User IDs who have written a review for each item
          item_ids = target_item_id, # Item IDs for which you want to predict ratings
          item_reviews = item_reviews, # Item reviews from the training set
          item_ids_of_reviews = review_item_ids, # Item IDs for which a user has written a review
          target_ratings = target_ratings, # Ground truth ratings
          target_reviews = target_reviews,
          num_of_user_item_pairs = num_of_user_item_pairs) # Ground truth reviews

    user_ids_stack = torch.stack(target_user_id, dim=0).squeeze().type(torch.LongTensor)
    item_ids_stack = torch.stack(target_item_id, dim=0).squeeze().type(torch.LongTensor)

    # Updated embeddings
    user_embeddings = narre.user_net.entity_id_embeddings(user_ids_stack)
    item_embeddings = narre.item_net.entity_id_embeddings(item_ids_stack)
        
    # Combining the updated ID embeddings with the already calculated latent features
    user_features_updated = torch.cat((user_embeddings, user_features[:, config.id_embedding_size:]), dim = 1)
    item_features_updated = torch.cat((item_embeddings, item_features[:, config.id_embedding_size:]), dim = 1)
    
    # Review generation
    raw_probabilities = mrg(user_features_updated, item_features_updated, target_reviews, num_of_user_item_pairs)
    raw_probabilities = raw_probabilities.view(raw_probabilities.shape[0] * raw_probabilities.shape[1], -1)
    target_reviews = torch.stack(target_reviews, dim = 0)
    target_reviews = target_reviews.view(target_reviews.shape[0] * target_reviews.shape[2])

    converted_prob = view_generated_reviews(raw_probabilities)
    print(converted_prob)
    # print(encoded_reviews.shape)
    # print(model)