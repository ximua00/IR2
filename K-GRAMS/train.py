# Importing the libraries
import torch
import numpy as np
import config
import model

# Setting up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the models
narre = model.NARRE(embedding_size = config.word_embedding_size, vocab_size = config.vocab_size, out_channels = config.num_filters, filter_size = config.filter_size, review_length = config.max_review_length, user_id_size = config.num_users, item_id_size = config.num_items, id_embedding_size = config.id_embedding_size, hidden_size = config.hidden_size, latent_size = config.latent_size)
mrg = model.MRG(word_embedding_size = config.word_embedding_size, id_embedding_size = config.id_embedding_size, hidden_dim = config.hidden_size, latent_size = config.latent_size, vocab_size = config.vocab_size)

# Loss variables
mse_loss = torch.nn.MSELoss()
ce_loss = torch.nn.CrossEntropyLoss()

# Optimizer variables
narre_optimizer = torch.optim.Adam(narre.parameters(), lr = config.narre_learning_rate)
mrg_optimizer = torch.optim.Adam(mrg.parameters(), lr = config.mrg_learning_rate)

# For all epochs
for epoch in range(config.total_epochs):
    # Load batches from the data loader
    batches = data_loader(whatevs)
    # Assuming a batch will have 
    # user_reviews, item_reviews, review_user_ids, review_item_ids, target_user_id, target_item_id, ground-truth rating, ground-truth review
    for batch_num, batch in enumerate(batches):
        # Unpacking the batch variable
        user_reviews, item_reviews, review_user_ids = batch[0], batch[1], batch[2]
        review_item_ids, target_user_id, target_item_id = batch[3], batch[4], batch[5]
        target_ratings, target_reviews = batch[6], batch[7]

        # Number of user-item pairs
        num_of_user_item_pairs = len(user_reviews)
        
        # Obtaining the predicted ratings from NARRE
        predicted_ratings, user_features, item_features, user_item_features = narre(user_ids = target_user_id, # User IDs for which you want to predict ratings
          user_reviews = user_reviews, # User reviews from the training set
          user_ids_of_reviews = review_user_ids, # User IDs who have written a review for each item
          item_ids = target_item_id, # Item IDs for which you want to predict ratings
          item_reviews = item_reviews, # Item reviews from the training set
          item_ids_of_reviews = review_item_ids, # Item IDs for which a user has written a review
          target_ratings = target_ratings, # Ground truth ratings
          target_reviews = target_reviews, # Ground truth reviews
          num_of_user_item_pairs = num_of_user_item_pairs) # Number of user-item pairs
        
        # Calculate loss for narre
        stacked_target_rating = torch.stack(target_ratings, dim=0).view(1, -1)
        rating_pred_loss = mse_loss(predicted_ratings, stacked_target_rating).squeeze()
        
        # Zero grad optimizer
        narre_optimizer.zero_grad()

        # Backward computation
        rating_pred_loss.backward()

        # Updating the NARRE parameters
        narre_optimizer.step()
        
        # MRG model
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

        # Computing the cross entropy loss
        review_generation_loss = ce_loss(raw_probabilities, target_reviews)

        # Zero grad optimizer
        mrg_optimizer.zero_grad()

        # Backward computation
        review_generation_loss.backward()
        
        # Updation
        mrg_optimizer.step()

        # Generate reviews after every 100 batches
        if batch_num % 100 == 0:
            view_generated_reviews(raw_probabilities)

def view_generated_reviews(raw_probabilities):

    # Soft-maxing the raw probabilities
    raw_probabilities = torch.exp(raw_probabilities) / torch.sum(torch.exp(raw_probabilities), dim = 2)

    # Taking the max index
    max_indices = torch.argmax(raw_probabilities, dim = 2)

    # List of generated sentences
    # generated_sentences = []

    # Generating the sentences
    for user_item_pair_index in range(max_indices.shape[0]):
        generated_sentence = []
        for word_index in max_indices[user_item_pair_index]:
            generated_sentence.append(id_to_word(word_index))
        print(' '.join(generated_sentence))
        # generated_sentences.append(generated_sentence)

