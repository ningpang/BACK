import torch
import math
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model.utils import sort_batch_by_length, init_lstm, init_linear


class Embedding(nn.Module):
    def __init__(self, word_vec_mat, embedding_dim=50):
        super(Embedding, self).__init__()
        unk = torch.zeros(1, embedding_dim)
        blk = torch.zeros(1, embedding_dim)
        word_vec_mat = torch.from_numpy(word_vec_mat)
        self.word_embedding = nn.Embedding(word_vec_mat.shape[0] + 2, embedding_dim,
                                           padding_idx=word_vec_mat.shape[0] + 1)
        self.word_embedding.weight.data.copy_(torch.cat((word_vec_mat, unk, blk), 0))

        for p in self.parameters():
            p.requires_grad = False

        self.fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.fc3 = nn.Linear(embedding_dim, embedding_dim)
        self.fc4 = nn.Linear(embedding_dim, embedding_dim)
        self.fc5 = nn.Linear(embedding_dim, embedding_dim)
        self.w = nn.Parameter(torch.ones(embedding_dim))
        self.b = nn.Parameter(torch.ones(embedding_dim))
        self.drop = nn.Dropout(0.1)

    def norm(self, x):
        mean = x.mean(-1, keepdim= True)
        std = x.std(-1, keepdim=True)
        return self.w * (x-mean)/(std+1e-6) + self.b

    def feed(self, x):
        return self.fc5(self.drop(F.relu(self.fc4(x))))

    def refine(self, embedding, mask):
        E1 = self.fc1(embedding)
        E2 = self.fc2(embedding)
        E3 = self.fc3(embedding)
        mask = mask.unsqueeze(-1).float() @ mask.unsqueeze(-2).float()
        d_k = E1.size(-1)
        scores = E1 @ E2.transpose(-2, -1)/math.sqrt(d_k)
        E = F.softmax(scores, dim=-1) @ E3
        E_ = self.norm(embedding+E)
        E = self.norm(E_+self.feed(E_))
        return E

    def forward(self, inputs):
        word = inputs['word']
        mask = inputs['mask']
        word_embedding = self.word_embedding(word)
        word_embedding = self.refine(word_embedding, mask)
        return word_embedding

class BACK(nn.Module):
    def __init__(self, args, word_vec_mat, drop=True):
        super(BACK, self).__init__()
        self.args = args
        self.device = args.device
        self.embedding_size = args.embedding_size
        self.hidden_size = args.hidden_size
        self.neg_k = args.Neg_K
        self.embedding = Embedding(word_vec_mat, self.embedding_size)
        self.lstm_base = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size,
                                 bidirectional=True, batch_first=True)
        self.lstm_interact = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size,
                                     bidirectional=True, batch_first=True)
        self.att_projection = nn.Linear(self.hidden_size*8, 1)
        self.fc1 = nn.Linear(self.hidden_size*2, self.hidden_size*2)
        self.fc2 = nn.Linear(self.hidden_size*2, self.hidden_size*2)
        self.projection = nn.Linear(self.hidden_size*8, self.hidden_size)
        self.MLP = nn.Sequential(nn.Linear(self.hidden_size*16, self.hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_size, 1))
        self.drop = drop
        if self.drop:
            self.dropout = nn.Dropout(self.args.dropout)
        self.apply(self.weight_init)

    def weight_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            init_linear(m)
        elif classname.find('LSTM') != -1:
            init_lstm(m)

    def contextual_encoder(self, input):
        for key in input:
            input[key] = input[key].to(self.device)
        input_mask = (input['mask'] != 0).float()
        max_length = input_mask.long().sum(1).max().item()
        input_mask = input_mask[:, :max_length].contiguous()
        sequence_lengths = input_mask.long().sum(1)

        embedding = self.embedding(input)

        embedding_ = embedding[:, :max_length].contiguous()


        if self.drop:
            embedding_ = self.dropout(embedding_)

        sorted_inputs, sorted_sequence_lengths, restoration_indices, _ = sort_batch_by_length(embedding_,
                                                                                              sequence_lengths)
        packed_sequence_input = pack_padded_sequence(sorted_inputs,
                                                     sorted_sequence_lengths.cpu(),
                                                     batch_first=True)
        output, _ = self.lstm_base(packed_sequence_input)
        unpacked_sequence_tensor, _ = pad_packed_sequence(output, batch_first=True)
        unpacked_sequence_tensor = unpacked_sequence_tensor.index_select(0, restoration_indices)

        return unpacked_sequence_tensor, input_mask, max_length

    def interaction_encoder(self, input, mask):
        if self.drop:
            input = self.dropout(input)
        mask = mask.squeeze(2)

        sequence_lengths = mask.long().sum(1)

        sorted_inputs, sorted_sequence_lengths, restoration_indices, _ = sort_batch_by_length(input, sequence_lengths)

        packed_sequence_input = pack_padded_sequence(sorted_inputs,
                                                     sorted_sequence_lengths.cpu(),
                                                     batch_first=True)
        output, _ = self.lstm_interact(packed_sequence_input)
        unpacked_sequence_tensor, _ = pad_packed_sequence(output, batch_first=True)
        unpacked_sequence_tensor = unpacked_sequence_tensor.index_select(0, restoration_indices)

        return unpacked_sequence_tensor

    def bidirectional_attention(self, support_word, query_word, support_mask, query_mask):
        support_len = support_word.size(1)
        query_len = query_word.size(1)
        C1 = support_word @ query_word.transpose(1, 2)
        support_word_ = support_word.unsqueeze(2).expand(-1, -1, query_len, -1)
        query_word_ = query_word.unsqueeze(1).expand(-1, support_len, -1, -1)
        C_matrix = self.feature_fusion(support_word_, query_word_, 3)
        C2 = self.att_projection(C_matrix).squeeze(-1)

        C = C1 + support_mask * query_mask.transpose(1, 2) * 100
        return C

    def feature_fusion(self, x, y, dim):
        return torch.cat([x, y, torch.abs(x - y), x * y], dim)

    def max_mean_agg(self, support_word, query_word, support_mask, query_mask, K):

        max_query_tilde, _ = torch.max(query_word, 1)
        mean_query_tilde = torch.sum(query_word, 1) / torch.sum(query_mask, 1)
        query_output = torch.cat([max_query_tilde, mean_query_tilde], 1)

        support_tilde = support_word.view(support_word.size(0) // K, K, -1, self.hidden_size * 2)
        support_mask = support_mask.view(support_tilde.size(0), K, -1, 1)

        max_support_tilde, _ = torch.max(support_tilde, 2)
        mean_support_tilde = torch.sum(support_tilde, 2) / torch.sum(support_mask, 2)
        support_output = torch.cat([max_support_tilde, mean_support_tilde], 2)

        return support_output, query_output


    def forward(self, support, query, N, K, Q, train=True):

        # Contextual encoder layer
        support_word, support_mask, support_len = self.contextual_encoder(support)
        query_word, query_mask, query_len = self.contextual_encoder(query)
        batch_size = support_word.size(0) // (N * K)


        # Bi-directional attention layer
        support_word = support_word.view(batch_size, 1, N, K, support_len, self.hidden_size * 2)\
                                   .expand(batch_size, N * Q, N, K, support_len, self.hidden_size * 2).contiguous()\
                                   .view(batch_size * N * Q * N, K * support_len, self.hidden_size * 2)
        support_mask = support_mask.view(batch_size, 1, N, K, support_len)\
                                   .expand(batch_size, N * Q, N, K, support_len).contiguous()\
                                   .view(-1, K * support_len, 1)

        query_word = query_word.view(batch_size, N * Q, 1, query_len, self.hidden_size * 2)\
                               .expand(batch_size, N * Q, N, query_len, self.hidden_size * 2).contiguous()\
                               .view(batch_size * N * Q * N, query_len, self.hidden_size * 2)
        query_mask = query_mask.view(batch_size, N * Q, 1, query_len)\
                               .expand(batch_size, N * Q, N, query_len).contiguous()\
                               .view(-1, query_len, 1)

        # word-wise similarity
        attention_matrix = self.bidirectional_attention(support_word, query_word, support_mask, query_mask)
        # support_to_query and query_to_support
        support_word_ = F.softmax(attention_matrix, 2) @ query_word * support_mask
        query_word_ = F.softmax(attention_matrix.transpose(1, 2), 2) @ support_word * query_mask
        # feature fusion
        support_word_bar = self.feature_fusion(support_word, support_word_, 2)
        query_word_bar = self.feature_fusion(query_word, query_word_, 2)
        support_word_bar = torch.relu(self.projection(support_word_bar))
        query_word_bar = torch.relu(self.projection(query_word_bar))
        support_word_bar = support_word_bar.view(batch_size * N * Q * N * K, support_len, self.hidden_size)
        support_mask = support_mask.view(batch_size * N * Q * N * K, support_len, 1)


        # Model layer
        support_word_tilde = self.interaction_encoder(support_word_bar, support_mask)
        query_word_tilde = self.interaction_encoder(query_word_bar, query_mask)
        support_word_tilde, query_word_tilde = self.max_mean_agg(support_word_tilde, query_word_tilde, support_mask, query_mask, K)
        # get the prototypes
        all_query_word = query_word_tilde.unsqueeze(1).repeat(1, K, 1)
        E = self.MLP(self.feature_fusion(all_query_word, support_word_tilde, 2))
        prototypes = (support_word_tilde.transpose(1, 2) @ F.softmax(E, 1)).squeeze(2)


        # Output layer
        logits = self.MLP(self.feature_fusion(query_word_tilde, prototypes, 1))

        # Large margin loss
        L_margin = nn.TripletMarginLoss(margin=1.0, p=2)
        margin_loss = 0.0
        if train:
            # Negative sampling
            negative_support_space = support_word_tilde.reshape(batch_size*N*Q, N*K, self.hidden_size*4)
            negative_samples = []
            for i in range(N):
                indices = list(np.random.choice(list(set(list(range(0, N*K)))
                               -set(list(range(i*K, (i+1)*K)))), self.neg_k*K, True))
                negative_samples.append(negative_support_space[:, indices, :])
            negative_samples = torch.cat(negative_samples, 1)
            margin_loss = L_margin(support_word_tilde.unsqueeze(-1).repeat(1, 1, self.neg_k, 1).view(-1, 4*self.hidden_size),
                                   prototypes.unsqueeze(1).repeat(1, self.neg_k*K, 1).view(-1, 4*self.hidden_size),
                                   negative_samples.view(-1, 4*self.hidden_size))


        logits = logits.view(batch_size * N * Q, N)
        _, pred = torch.max(logits, 1)

        return logits, pred, margin_loss







