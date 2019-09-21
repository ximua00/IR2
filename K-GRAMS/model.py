import torch
import torch.nn as nn

class KGRAMS(nn.Module):
    def __init__(self, embedding_size, vocab_size, out_channels, filter_size, review_length, user_id_size, item_id_size, embedding_id_size, hidden_size)    :
        super(KGRAMS, self).__init__()
        self.word_embeddings = nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=embedding_size)
        self.user_net = EntityNet(embedding_id_size=embedding_id_size,
                                  out_channels=out_channels,
                                  filter_size=filter_size,
                                  review_length=review_length,
                                  score_size=item_id_size,
                                  id_size=user_id_size,
                                  hidden_size=hidden_size)
        self.item_net = EntityNet(embedding_id_size=embedding_id_size,
                                  out_channels=out_channels,
                                  filter_size=filter_size,
                                  review_length=review_length,
                                  score_size=user_id_size,
                                  id_size=item_id_size,
                                  hidden_size=hidden_size)


    def forward(self, target_user_id, user_revs, review_user_ids, target_item_id, item_revs, review_item_ids):
        u = self.user_net(target_user_id, user_revs, review_item_ids, self.word_embeddings)
        i = self.item_net(target_item_id, item_revs, review_user_ids, self.word_embeddings)


class EntityNet(nn.Module):
    def __init__(self, embedding_id_size, out_channels, filter_size, review_length, score_size, id_size, hidden_size):
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


    def forward(self, target_id, reviews, review_ids, word_embeddings):
        attentions = []
        features = []
        for review, review_id in zip(reviews, review_ids):
            rev_embedding = word_embeddings(review)
            input_embeddings = rev_embedding.view(-1, 1, rev_embedding.size(1), rev_embedding.size(2))
            x = self.conv2d(input_embeddings)
            x = self.relu(x)
            review_feats = self.max_pool(x).squeeze() #4D : batch_size(1) * out_channels * height(1) * width(1)
            review_feats = review_feats.view(-1, 1)
            score_embedding = self.entity_score_embeddings(review_id).view(-1, 1)
            review_attention = torch.mm(self.h.view(1, -1), self.relu(torch.mm(self.W_O, review_feats) + torch.mm(self.W_u, score_embedding) + self.b1.view(-1,1))) + self.b2
            features.append(review_feats)
            attentions.append(review_attention)
            
        return None

if __name__ == "__main__":
    item_reviews = [torch.randint(low=0, high=2, size=(1,80)), torch.randint(low=0, high=2, size=(1,80))]
    user_reviews = [torch.randint(low=0, high=2, size=(1,80)), torch.randint(low=0, high=2, size=(1,80))]

    review_user_ids = [torch.randint(low=0, high=1, size=(1,1)), torch.randint(low=0, high=1, size=(1,1))]
    review_item_ids = [torch.randint(low=0, high=2, size=(1, 1)), torch.randint(low=0, high=2, size=(1, 1))]

    target_user_id = torch.randint(low=0, high = 1, size=(1,1))
    target_item_id = torch.randint(low=0, high=2, size=(1,1))


    model = KGRAMS(embedding_size = 100, vocab_size = 3 , out_channels = 100, filter_size = 3, review_length=80, user_id_size = 2, item_id_size = 3, embedding_id_size = 100, hidden_size = 50)

    model(target_user_id = target_user_id,
          user_revs = user_reviews,
          review_user_ids = review_user_ids,
          target_item_id=target_item_id,
          item_revs = item_reviews,
          review_item_ids = review_item_ids)

    # print(model)