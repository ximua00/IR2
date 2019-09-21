import torch
import torch.nn as nn

class KGRAMS(nn.Module):
    def __init__(self, embedding_size, vocab_size, out_channels, filter_size, review_length, user_id_size, item_id_size, embedding_id_size, hidden_size)    :
        super(KGRAMS, self).__init__()
        self.word_embeddings = nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=embedding_size)
        self.user_net = EntityNet(embedding_id_size, out_channels, filter_size, review_length, item_id_size, user_id_size, hidden_size)
        self.item_net = EntityNet(embedding_id_size, out_channels, filter_size, review_length, user_id_size, item_id_size, hidden_size)


    def forward(self, user_id, user_rev, item_id, item_rev):
        user_rev_embedding = self.word_embeddings(user_rev)
        item_rev_embedding = self.word_embeddings(item_rev)
        u = self.user_net(user_rev_embedding, item_id)
        i = self.item_net(item_rev_embedding, user_id)


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


    def forward(self, input, score_id):
        input_embeddings = input.view(-1, 1, input.size(1), input.size(2))
        x = self.conv2d(input_embeddings)
        x = self.relu(x)
        review_feats = self.max_pool(x).squeeze() #4D : batch_size(1) * out_channels * height(1) * width(1)
        review_feats = review_feats.view(-1, 1)
        score_embeddings = self.entity_score_embeddings(score_id).view(-1, 1)

        review_attention = torch.mm(self.h.view(1, -1), self.relu(torch.mm(self.W_O, review_feats) + torch.mm(self.W_u, score_embeddings) + self.b1.view(-1,1))) + self.b2

        return review_attention

if __name__ == "__main__":
    item_review = torch.randint(low=0, high=2, size=(1,80))
    user_review = torch.randint(low=0, high=2, size=(1, 80))
    user_id = torch.randint(low=0, high = 1, size=(1,1))
    item_id = torch.randint(low=0, high=2, size=(1,1))

    model = KGRAMS(embedding_size = 100, vocab_size = 3 , out_channels = 100, filter_size = 3, review_length=80, user_id_size = 2, item_id_size = 3, embedding_id_size = 100, hidden_size = 50)
    model(user_id, user_review, item_id, item_review)