import sys, os
import collections
import zipfile
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch import optim
import numpy as np

'''
    EncoderDecoder framework
'''
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


'''
    loss function
'''
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def SequenceMask(self, X, X_len,value=0):
        maxlen = X.size(1)
        mask = torch.arange(maxlen)[None, :].to(X_len.device) < X_len[:, None]   
        X[~mask]=value
        return X

    def forward(self, pred, label, valid_length):
        # the sample weights shape should be (batch_size, seq_len)
        weights = torch.ones_like(label)
        weights = self.SequenceMask(weights, valid_length).float()
        self.reduction='none'
        output=super(MaskedSoftmaxCELoss, self).forward(pred.transpose(1,2), label)
        return (output*weights).mean(dim=1)

'''
    attention
'''
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


'''
    Seq2Seq model with Attention
'''   
class AttenSeq2SeqDecoder(Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(AttenSeq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)
        self.atten = MLPAttention(num_hiddens, num_hiddens, dropout)

    def init_state(state, valid_len, *args):
        enc_outputs, hidden_state = state
        return [enc_outputs.permute(1, 0, -1), hidden_state, valid_len]

    def forward(self, X, state):
        enc_outputs, hidden_state, valid_len = state
        X = self.embedding(X).transpose(0, 1)
        outputs = []
        for x in X:
            query = hidden_state[0][-1].unsqueeze(1)
            key      = enc_outputs
            value  = enc_outputs
            context = self.atten(query, key, value, valid_len)
            x = torch.cat((context, x.unsqueeze(1)), dim = -1)
            out, hidden_state = self.rnn(x.transpose(0, 1), hidden_state)
            outputs.append(out)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.transpose(0, 1), [enc_outputs, hidden_state, valid_len]


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
    
    
def SequenceMask(X, X_len,value=-1e6):
    maxlen = X.size(1)
    X_len = X_len.to(X.device)
    #print(X.size(),torch.arange((maxlen),dtype=torch.float)[None, :],'\n',X_len[:, None] )
    mask = torch.arange((maxlen), dtype=torch.float, device=X.device)
    mask = mask[None, :] < X_len[:, None]
    #print(mask)
    X[~mask]=value
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
        # fill masked elements with a large negative, whose exp is 0
        X = SequenceMask(X.reshape((-1, shape[-1])), valid_length)
 
        return softmax(X).reshape(shape)



'''
    Transformer
'''
def transpose_qkv(X, num_heads):
    '''shape (batch_size, seq_len, num_heads * hidden_size) to (batch_size * num_heads, seq_len, hidden_size)'''
    X = X.view(X.shape[0], X.shape[1], num_heads, -1)
    X = X.transpose(2, 1).contiguous()
    output = X.view(-1, X.shape[2], X.shape[3])
    return output

def transpose_output(X, num_heads):
    '''the reverse operation of transpose_qkv'''
    X = X.view(-1, num_heads, X.shape[1], X.shape[2])
    X = X.transpose(2, 1).contiguous()
    return X.view(X.shape[0], X.shape[1], -1)

def handle_valid_length(valid_length, num_heads):
    # Copy valid_length by num_heads times
    device = valid_length.device
    valid_length = valid_length.cpu().numpy() if valid_length.is_cuda else valid_length.numpy()
    if valid_length.ndim == 1:
        valid_length = torch.FloatTensor(np.tile(valid_length, num_heads))
    else:
        valid_length = torch.FloatTensor(np.tile(valid_length, (num_heads,1)))
    valid_length = valid_length.to(device)
    return valid_length

'''
    多头注意力机制
'''
class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, dropout, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.wq = nn.Linear(input_size, hidden_size, bias=False)
        self.wk = nn.Linear(input_size, hidden_size, bias=False)
        self.wv = nn.Linear(input_size, hidden_size, bias=False)
        self.wo = nn.Linear(hidden_size, hidden_size, bias=False)
        
    def forward(self, query, key, value, valid_length):
        query = transpose_qkv(self.wq(query), self.num_heads)
        key      = transpose_qkv(self.wk(key), self.num_heads)
        value  = transpose_qkv(self.wv(value), self.num_heads)
        if valid_length is not None:
            valid_length = handle_valid_length(valid_length, self.num_heads)
        output = self.attention(query, key, value, valid_length)
        output_concat = transpose_output(output, self.num_heads)
        return self.wo(output_concat)


'''
    基于位置的前馈网络
'''
class PositionWiseFFN(nn.Module):
    def __init__(self, input_size, ffn_hidden_size, hidden_size_out, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.ffn_1 = nn.Linear(input_size, ffn_hidden_size)
        self.ffn_2 = nn.Linear(ffn_hidden_size, hidden_size_out)
    
    def forward(self, X):
        return self.ffn_2(F.relu(self.ffn_1(X)))

'''
    Add and Norm
'''
class AddNorm(nn.Module):
    def __init__(self, hidden_size, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, X, Y):
        return self.norm(self.dropout(Y) + X)
    
'''
    位置编码
'''
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = np.zeros((1, max_len, embed_size))
        X = np.arange(0, max_len).reshape(-1, 1) / np.power(
            10000, np.arange(0, embed_size, 2)/embed_size)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)
        self.P = torch.FloatTensor(self.P)
        
    def forward(self, X):
        if X.is_cuda and not self.P.is_cuda:
            self.P = self.P.cuda()
        X = X + self.P[:, :X.shape[1], :]
        return self.dropout(X)
    
'''
    Encoder Block
'''
class EncoderBlock(nn.Module):
    def __init__(self, embed_size, ffn_hidden_size, num_heads, dropout, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(embed_size, embed_size, num_heads, dropout)
        self.add_norm1 = AddNorm(embed_size, dropout)
        self.ffn = PositionWiseFFN(embed_size, ffn_hidden_size, embed_size)
        self.add_norm2 = AddNorm(embed_size, dropout)
    
    def forward(self, X, valid_length):
        Y = self.add_norm1(X, self.attention(X, X, X, valid_length))
        return self.add_norm2(Y, self.ffn(Y))

'''
    Transformer Encoder
'''
class TransformerEncoder(Encoder):
    def __init__(self, vocab_size, embed_size, ffn_hidden_size, num_heads, num_layers, dropout, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoding = PositionalEncoding(embed_size, dropout)
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append( EncoderBlock(embed_size, ffn_hidden_size, num_heads, dropout))
            
    def forward(self, X, valid_length, *args):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.embed_size))
        for block in self.blocks:
            X = block(X, valid_length)
        return X

'''
    Transformer Decoder block
'''
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, ffn_hidden_size, num_heads, dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.atten1 = MultiHeadAttention(embed_size, embed_size, num_heads, dropout)
        self.add_norm1 = AddNorm(embed_size, dropout)
        self.atten2 = MultiHeadAttention(embed_size, embed_size, num_heads, dropout)
        self.add_norm2 = AddNorm(embed_size, dropout)
        self.ffn = PositionWiseFFN(embed_size, ffn_hidden_size, embed_size)
        self.add_norm3 = AddNorm(embed_size, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_length = state[0], state[1]
        if state[2][self.i] is None:
            key_value = X
        else:
            key_value = torch.cat((state[2][self.i], X), dim=1)
        state[2][self.i] = key_value

        if self.training:
            batch_size, seq_len, _ = X.shape
            valid_length = torch.FloatTensor(np.tile(np.arange(1, seq_len+1), (batch_size, 1)))
            valid_length = valid_length.to(X.device)
        else:
            valid_length = None
        X2 = self.atten1(X, key_value, key_value, valid_length)
        Y = self.add_norm1(X, X2)
        Y2 = self.atten2(Y, enc_outputs, enc_outputs, enc_valid_length)
        Z = self.add_norm2(Y, Y2)
        return self.add_norm3(Z, self.ffn(Z)), state

class TransformerDecoder(Decoder):
    def __init__(self, vocab_size, embed_size, ffn_hidden_size, num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoding = PositionalEncoding(embed_size, dropout)
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(DecoderBlock(embed_size, ffn_hidden_size, num_heads, dropout, i))
        self.dense = nn.Linear(embed_size, vocab_size)

    def init_state(self, enc_outputs, enc_valid_length, *args):
        return [enc_outputs, enc_valid_length, [ None for i in range(self.num_layers)] ]
    
    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.embed_size))
        for block in self.blocks:
            X, state = block(X, state)
        return self.dense(X), state


