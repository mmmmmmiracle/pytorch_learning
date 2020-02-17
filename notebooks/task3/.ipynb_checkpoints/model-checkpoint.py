import sys, os
import collections
import zipfile
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch import optim


class Encoder(nn.Module):
    def __init__(self,**kwargs):
        super(Encoder, self).__init__(**kwargs)
    
    def forward(self, X, *args):
        raise NotImplementedError
    
class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)
    
    def init_state(self, encoded_state, *args):
        raise NotImplementedError
        
    def forward(self, X, state):
        raise NotImplementedError

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, enc_X, dec_X, *args):
        state = self.encoder(enc_X, *args)
        decoded_state = self.decoder.init_state(state, *args)
        return self.decoder(dec_X, decoded_state)

class Seq2SeqEncoder(Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, num_hiddens, num_layers, dropout=dropout)
    
    def begin_state(self, batch_size, device):
        H = torch.zeros(size=(self.num_layers, batch_size, self.num_hiddens),  device=device)
        C = torch.zeros(size=(self.num_layers, batch_size, self.num_hiddens),  device=device)
        return [H, C]
    
    def forward(self, X, *args):
        X = self.embedding(X)
        X = X.transpose(0, 1)
        out, state = self.rnn(X)
        return out, state
        
class Seq2SeqDecoder(Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)
        
    def init_state(self, state, *args):
        return state[1]
    
    def forward(self, X, state):
        X = self.embedding(X).transpose(0, 1)
        out, state = self.rnn(X, state)
        out = self.dense(out).transpose(0, 1)
        return out, state

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label, device = label.device)
        weights = SequenceMask(weights, valid_len).float()
        self.reduction = 'none'
        output = super(MaskedSoftmaxCELoss, self).forward(pred.transpose(1,2), label)
        return (output * weights).mean(dim=1)


class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, valid_len = None):
        d = query.shape[-1]
        scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(d)
        weights = self.dropout(masked_softmax(scores, valid_len))
#         print(weights)
        return torch.bmm(weights, value)

class MLPAttention(nn.Module):
    def __init__(self, input_dim, units, dropout, **kwargs):
        super(MLPAttention, self).__init__(**kwargs)
        self.wk = nn.Linear(input_dim, units, bias=False)
        self.wq = nn.Linear(input_dim, units, bias=False)
        self.v     = nn.Linear(units, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, valid_len):
        query, key = self.wq(query), self.wk(key)
        features = query.unsqueeze(2) + key.unsqueeze(1)
        scores = self.v(features).squeeze(-1)
        weights = self.dropout(masked_softmax(scores, valid_len))
        return torch.bmm(weights, value)
    
class MLPAttention_v2(nn.Module):
    def __init__(self, input_dim, units, dropout, **kwargs):
        super(MLPAttention_v2, self).__init__(**kwargs)
        self.wk = nn.Linear(input_dim, units, bias=False)
        self.wq = nn.Linear(input_dim, units, bias=False)
        self.v     = nn.Linear(units + units, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, valid_len):
        query, key = self.wq(query), self.wk(key)
        features = torch.cat((query.repeat(1, key.shape[1] ,1), key), dim=-1)
        print(features.shape)
        scores = self.v(features)
        weights = self.dropout(masked_softmax(scores, valid_len))
        print(weights.shape, value.shape)
        return torch.bmm(weights.transpose(1, 2), value)

        
def grad_clipping(params, theta, device):
    """Clip the gradient."""
    norm = torch.tensor([0], dtype=torch.float32, device=device)
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data.mul_(theta / norm)

def grad_clipping_nn(model, theta, device):
    """Clip the gradient for a nn model."""
    grad_clipping(model.parameters(), theta, device)
    
    
def SequenceMask(X, X_len,value=0):
    maxlen = X.size(1)
    #print(X.size(),torch.arange((maxlen),dtype=torch.float)[None, :],'\n',X_len[:, None] )
#     print(X_len.device, X.device)
    mask = torch.arange((maxlen),dtype=torch.float, device=X.device)[None, :] >= X_len[:, None]   
    #print(mask)
    X[mask]=value
    return X
    
def masked_softmax(X, valid_length):
    # X: 3-D tensor, valid_length: 1-D or 2-D tensor
    softmax = nn.Softmax(dim=-1)
    if valid_length is None:
        return softmax(X)
    else:
        shape = X.shape
        if valid_length.dim() == 1:
            try:
                valid_length = torch.FloatTensor(valid_length.numpy().repeat(shape[1], axis=0))#[2,2,3,3]
            except:
                valid_length = torch.FloatTensor(valid_length.cpu().numpy().repeat(shape[1], axis=0))#[2,2,3,3]
        else:
            valid_length = valid_length.reshape((-1,))
#         print(valid_length.device)
        # fill masked elements with a large negative, whose exp is 0
        X = SequenceMask(X.reshape((-1, shape[-1])), valid_length.to(X.device))
 
        return softmax(X).reshape(shape)