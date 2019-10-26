
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import sys
import config
import csv

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = config.device

class BiLSTMEncoder(nn.Module):

    def __init__(self, embed_dim,hidden_dim,layers,dropout_lstm, dropout_input=0.2):
        super(BiLSTMEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.dropout_input = dropout_input
        self.dropout_lstm = dropout_lstm
        self.rnn = nn.LSTM(input_size = embed_dim, #512
                           hidden_size = hidden_dim, #hyper
                           num_layers = layers, #1
                           dropout = dropout_lstm, 
                           bidirectional = True,
                           batch_first = True)
        self.input_dropout = nn.Dropout(dropout_input)
    def forward(self, inputs, lengths):
        embedded_input = self.input_dropout(inputs)
        # (sorted_input, sorted_lengths, input_unsort_indices, _) = sort_batch_by_length(embedded_input, lengths)
        # packed_input = pack_padded_sequence(sorted_input, sorted_lengths.data.tolist(), batch_first=True)
        # print(lengths.data.tolist())
        lengths = torch.as_tensor(lengths.cpu(), dtype = torch.int64)
        lengths = lengths.squeeze()
        packed_input = pack_padded_sequence(embedded_input, lengths, batch_first=True)
        # if torch.cuda.is_available():
        #     packed_input = packed_input.to(device=torch.device('cuda'))
        embedding, _ = self.rnn(packed_input)
        embedding, _ = pad_packed_sequence(embedding, batch_first=True)
        # embedding = embedding[input_unsort_indices]
        return embedding

class lin_softmax(nn.Module):
    def __init__(self, dropout, num_classes, hidden_dim):
        super(lin_softmax, self).__init__()
        self.fcl = nn.Linear(hidden_dim*2, num_classes)
        self.linear_dropout = nn.Dropout(dropout)

    def forward(self, output):

        input_encoding = self.linear_dropout(output)
        unnormalized_output = self.fcl(input_encoding)
        normalized_output = F.log_softmax(unnormalized_output, dim=-1)

        return normalized_output

# Self-attention layer from https://gist.github.com/cbaziotis/94e53bdd6e4852756e0395560ff38aa4
class SelfAttention(nn.Module):
    def __init__(self, attention_size,
                 batch_first=True,
                 layers=1,
                 dropout=.0,
                 non_linearity="tanh"):
        super(SelfAttention, self).__init__()

        self.batch_first = batch_first

        if non_linearity == "relu":
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()

        modules = []
        for i in range(layers - 1):
            modules.append(nn.Linear(attention_size, attention_size))
            modules.append(activation)
            modules.append(nn.Dropout(dropout))

        # last attention layer must output 1
        modules.append(nn.Linear(attention_size, 1))
        modules.append(activation)
        modules.append(nn.Dropout(dropout))

        self.attention = nn.Sequential(*modules)

        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def get_mask(attentions, lengths):
        """
        Construct mask for padded itemsteps, based on lengths
        """
        max_len = max(lengths.data)
        mask = Variable(torch.ones(attentions.size())).detach()
        mask = mask.to(device)

        for i, l in enumerate(lengths.data):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0
        return mask

    def forward(self, inputs, lengths):

        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        scores = self.attention(inputs).squeeze()
        scores = self.softmax(scores)

        ##################################################################
        # Step 2 - Masking
        ##################################################################

        # construct a mask, based on sentence lengths
        mask = self.get_mask(scores, lengths)

        # apply the mask - zero out masked timesteps
        masked_scores = scores * mask

        # re-normalize the masked scores
        _sums = masked_scores.sum(-1, keepdim=True)  # sums per row
        scores = masked_scores.div(_sums)  # divide by row sum

        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        # sum the hidden states
        representations = weighted.sum(1).squeeze()

        return representations, scores, weighted

def sort_batch_by_length(tensor: torch.Tensor, sequence_lengths: torch.Tensor):
    """
    Sort a batch first tensor by some specified lengths.
    Parameters
    ----------
    tensor : torch.FloatTensor, required.
        A batch first Pytorch tensor.
    sequence_lengths : torch.LongTensor, required.
        A tensor representing the lengths of some dimension of the tensor which
        we want to sort by.
    Returns
    -------
    sorted_tensor : torch.FloatTensor
        The original tensor sorted along the batch dimension with respect to sequence_lengths.
    sorted_sequence_lengths : torch.LongTensor
        The original sequence_lengths sorted by decreasing size.
    restoration_indices : torch.LongTensor
        Indices into the sorted_tensor such that
        ``sorted_tensor.index_select(0, restoration_indices) == original_tensor``
    permutation_index : torch.LongTensor
        The indices used to sort the tensor. This is useful if you want to sort many
        tensors using the same ordering.
    """

    if not isinstance(tensor, torch.Tensor) or not isinstance(sequence_lengths, torch.Tensor):
        raise ConfigurationError("Both the tensor and sequence lengths must be torch.Tensors.")

    sequence_lengths = sequence_lengths[((sequence_lengths != 0).nonzero()).squeeze()]
    sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
    permutation_index = permutation_index.to(device = sequence_lengths.device)
    sorted_tensor = tensor.index_select(0, permutation_index.long())
    # print(sorted_tensor)
    # sorted_tensor = torch.index_select(tensor, 0, permutation_index)
    index_range = torch.arange(0, len(sequence_lengths), device = sequence_lengths.device)
    # This is the equivalent of zipping with index, sorting by the original
    # sequence lengths and returning the now sorted indices.
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)
    return sorted_tensor, sorted_sequence_lengths, restoration_indices, permutation_index

def normalize(x):
    x_normed = (x-  x.min(0, keepdim=True)[0]) / (x.max(0, keepdim=True)[0]-  x.min(0, keepdim=True)[0])
    return x_normed
    
class main_model(nn.Module):
    

    def __init__(self, embed_dim, hidden_dim, layers, dropout_lstm, dropout_input, dropout_FC, dropout_lstm_2, dropout_input_2, dropout_attention, batch_size):
       
        super(main_model, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.dropout_FC = dropout_FC
        self.dropout_lstm = dropout_lstm
        self.dropout_input = dropout_input
        self.embedding = BiLSTMEncoder(embed_dim, hidden_dim, layers, dropout_lstm, dropout_input)
        self.self_attention = SelfAttention(2*hidden_dim, dropout_attention)
        self.self_attention_sentence = SelfAttention(2*hidden_dim, dropout_attention)
        self.doc_embedding = BiLSTMEncoder(2*hidden_dim, hidden_dim, layers, dropout_lstm_2, dropout_input_2)
        self.batch_size = batch_size
        # self.l_softmax = lin_softmax(dropout_FC, num_classes, hidden_dim)
        # self.doc_classifier = lin_softmax(dropout_FC, num_classes, hidden_dim)
        if torch.cuda.is_available():
            self.embedding.to(device = torch.device('cuda'))
            # self.l_softmax.to(device=torch.device('cuda'))
    
    def forward(self, rev_embeddings, num_of_reviews, mode):

        if torch.cuda.is_available():
            rev_embeddings = rev_embeddings.to(device=torch.device('cuda'))

        rev_lengths = torch.ones(rev_embeddings.shape[0], 1) * rev_embeddings.shape[1]
        
        # First BiLSTM
        encoded_words = self.embedding(rev_embeddings, rev_lengths) 
        averaged_sentences, attention, weighted = self.self_attention_sentence(encoded_words, rev_lengths.int())
        if(mode == 'test'):
            sigm = nn.Sigmoid()
            soft = nn.Softmax()
            rev_lengths = rev_lengths.to(device)
            attention_list = normalize(soft(rev_lengths * attention)).tolist()
        # shape of attention list : 640 X 82 (32*20 reviews with 82 words per review)

        doc_lengths = torch.ones(self.batch_size, 1) * num_of_reviews
        doc_lengths = doc_lengths.squeeze()

        predicted_docs = torch.split(averaged_sentences, split_size_or_sections = num_of_reviews)
        predicted_docs = torch.stack(predicted_docs, dim = 0)
        
        # 2nd BiLSTM
        out_encoding = self.doc_embedding.forward(predicted_docs, doc_lengths) 
        averaged_docs, attention, weighted = self.self_attention(out_encoding, doc_lengths)
        if(mode == 'test'):
            attention_list_sen = normalize(sigm(attention)).tolist()
            attention_list_sen_flat = [item for sublist in attention_list_sen for item in sublist]
            # print("Shape of flattened attention sentences", len(attention_list_sen_flat))
            # shape of attention_list_sen : 32 * 20  (32 data-points, 20 reviews per user(datapoint))
            with open('attention_stuff.csv', 'w') as csvfile:
              writer = csv.writer(csvfile, delimiter='\n')
              for i, j in zip(attention_list, attention_list_sen_flat):
                  writer.writerow((str(i), j))
        return averaged_docs 
