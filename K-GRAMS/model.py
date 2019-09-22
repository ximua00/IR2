import torch
import torch.nn as nn

class KGRAMS(nn.Module):
    def __init__(self, embedding_size, vocab_size, out_channels, filter_size, review_length, user_id_size, item_id_size, embedding_id_size, hidden_size, latent_size)    :
        super(KGRAMS, self).__init__()
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


    def forward(self, user_ids, user_reviews, user_ids_of_reviews, item_ids, item_reviews, item_ids_of_reviews):
        #Iterating through reviews of each user
        for index, target_user_id in enumerate(user_ids):
            one_user_revs = user_reviews[index]  #Tensor of size: #reviews X review_length
            review_item_ids = item_ids_of_reviews[index] #Tensor of size: #reviews X 1
            u = self.user_net(target_user_id, one_user_revs, review_item_ids, self.word_embeddings)

        # Iterating through reviews of each item
        for index, target_item_id in enumerate(item_ids):
            one_item_revs = item_reviews[index]  #Tensor of size: #reviews X review_length
            review_user_ids = user_ids_of_reviews[index] #Tensor of size: #reviews
            i = self.item_net(target_item_id, one_item_revs, review_user_ids, self.word_embeddings)


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
        batch_size = reviews.size(0)
        rev_embeddings = word_embeddings(reviews)   # o/p size: #reviews X review_length X embedding_size
        input_embeddings = rev_embeddings.view(rev_embeddings.size(0), 1, rev_embeddings.size(1), rev_embeddings.size(2))
        x = self.conv2d(input_embeddings)  #o/p size:  batch_size * out_channels * filter_out_size1 * filter_out_size2
        x = self.relu(x)
        review_feats = self.max_pool(x).squeeze().t() #4D : batch_size * out_channels * 1 * 1
        score_embedding = self.entity_score_embeddings(review_ids).view(-1, batch_size)
        review_attentions = torch.mm(self.h.view(1, -1), self.relu(torch.mm(self.W_O, review_feats) + torch.mm(self.W_u, score_embedding) + self.b1.view(-1,1))) + self.b2
        review_attentions = self.softmax(review_attentions)  #1 X batch_size
        reviews_importance = torch.mm(review_attentions, review_feats.t() )
        entity_features = self.linear(reviews_importance)
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
    batch_size=9
    item_reviews = [torch.randint(low=0, high=2, size=(batch_size,80)), torch.randint(low=0, high=2, size=(batch_size,80))]
    user_reviews = [torch.randint(low=0, high=2, size=(batch_size,80)), torch.randint(low=0, high=2, size=(batch_size,80))]

    review_user_ids = [torch.randint(low=0, high=1, size=(batch_size,1)), torch.randint(low=0, high=1, size=(batch_size,1))]
    review_item_ids = [torch.randint(low=0, high=2, size=(batch_size, 1)), torch.randint(low=0, high=2, size=(batch_size, 1))]

    target_user_id = [torch.randint(low=0, high = 1, size=(1,1)), torch.randint(low=0, high = 1, size=(1,1))]
    target_item_id = [torch.randint(low=0, high=2, size=(1,1)), torch.randint(low=0, high=2, size=(1,1))]

    model = KGRAMS(embedding_size = 100, vocab_size = 3 , out_channels = 100, filter_size = 3, review_length=80, user_id_size = 2, item_id_size = 3, embedding_id_size = 100, hidden_size = 50, latent_size=70)

    model(user_ids = target_user_id,
          user_reviews = user_reviews,
          user_ids_of_reviews = review_user_ids,
          item_ids = target_item_id,
          item_reviews = item_reviews,
          item_ids_of_reviews = review_item_ids)

    # print(model)