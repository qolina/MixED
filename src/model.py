import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from util import init_embedding, tensor2var

class TrigDecoderw(nn.Module):
    def __init__(self, config):
        super(TrigDecoderw, self).__init__()
        self.config = config
        hidden_dim_encoder = 2*config.lstm_hidden_dim if config.bilstm else config.lstm_hidden_dim
        self.hidden2tag = nn.Linear(hidden_dim_encoder, config.tagset_size)
        self.softmax = nn.LogSoftmax(dim=1)
        if config.gpu: self = self.cuda()

    # hidden_in shape: (batch_size, sent_length, encoder_hidden_dim)
    def forward(self, hidden_in, batch_sent_lens):
        (bsize, slen, hd) = hidden_in.size()
        hidden_in = hidden_in.view(bsize*slen, -1)
        tag_space = self.hidden2tag(hidden_in)
        tag_space = self.softmax(tag_space)
        return tag_space


class TrigDecoder(nn.Module):
    def __init__(self, config):
        super(TrigDecoder, self).__init__()
        self.config = config

        hidden_dim_encoder = 2*config.lstm_hidden_dim if config.bilstm else config.lstm_hidden_dim

        self.hidden2tag = nn.Linear(hidden_dim_encoder, config.tagset_size)
        self.hidden2biotag = nn.Linear(hidden_dim_encoder, 2*config.tagset_size-1)
        if config.gpu: self = self.cuda()

    # hidden_in shape: (batch_size, sent_length, encoder_hidden_dim)
    def forward(self, hidden_in, batch_sent_lens):
        (bsize, slen, hd) = hidden_in.size()
        hidden_in = hidden_in.view(bsize*slen, -1)
        tag_space = self.hidden2tag(hidden_in)
        bio_tag_space = self.hidden2biotag(hidden_in)
        return tag_space, bio_tag_space

class RNNEncoder(nn.Module):
    def __init__(self, config):
        super(RNNEncoder, self).__init__()
        self.config = config
        if config.random_seed != -1:
            torch.manual_seed(config.random_seed)
            np.random.seed(config.random_seed)
        self.num_directions = 2 if config.bilstm else 1
        self.lstm_in_dim = config.embed_dim

        self.word_embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
        self.drop = nn.Dropout(config.dropout)
        if self.config.gru:
            self.rnn = nn.GRU(self.lstm_in_dim, config.lstm_hidden_dim, num_layers=config.num_layers, bidirectional=config.bilstm)
        else:
            self.rnn = nn.LSTM(self.lstm_in_dim, config.lstm_hidden_dim, num_layers=config.num_layers, bidirectional=config.bilstm)

        if config.gpu: self = self.cuda()

    def forward(self, batch, batch_sent_lens, wpos, hidden=None):
        '''
            Input:
                batch: (batch_size, max_sent_len)
                batch_sent_lens: (batch_size, )
            Output:
                hidden_in, (batch_size, max_sent_len, hidden_dim)
        '''
        (bsize, slen) = batch.size()

        embeds = self.word_embeddings(batch) # size: batch_size*sent_length*word_embed_size
        embeds = self.drop(embeds)

        # because lstm_in, lstm_out: sent_length * batch_size * hidden_dim
        embeds = embeds.transpose(0, 1)
        embeds_pack = pack_padded_sequence(embeds, batch_sent_lens.numpy())
        lstm_out, hidden = self.rnn(embeds_pack, hidden)
        hidden_in, len_batch = pad_packed_sequence(lstm_out)

        hidden_in = hidden_in.transpose(0, 1).contiguous() # batch_size * sent_length * hidden_dim

        return hidden_in

