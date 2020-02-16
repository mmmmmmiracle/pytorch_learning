#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

import sys
sys.path.append("../input/")
import d2l_jay9460 as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()


# ## GRU
# 
# $$
# R_{t} = σ(X_tW_{xr} + H_{t−1}W_{hr} + b_r)\\    
# Z_{t} = σ(X_tW_{xz} + H_{t−1}W_{hz} + b_z)\\  
# \widetilde{H}_t = tanh(X_tW_{xh} + (R_t ⊙H_{t−1})W_{hh} + b_h)\\
# H_t = Z_t⊙H_{t−1} + (1−Z_t)⊙\widetilde{H}_t
# $$

# In[4]:


class GRU():
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_outputs = num_outputs
        
    def get_params(self):
        def _set_W(shape):
            W = torch.tensor(np.random.normal(0, 0.01, size=shape), device = device, dtype = dtype)
            return nn.Parameter(W, requires_grad=True)
            
        def set_param():
            W_x = _set_W(shape = (self.num_inputs, self.num_hiddens))
            W_h = _set_W(shape = (self.num_hiddens, self.num_hiddens))
            b   = nn.Parameter(torch.zeros(self.num_hiddens, device=device, dtype=dtype), requires_grad=True)
            return (W_x, W_h, b)
        
        W_xr, W_hr, b_r = set_param()
        W_xz, W_hz, b_z = set_param()
        W_xh, W_hh, b_h = set_param()
        
        W_ho = _set_W(shape = (self.num_hiddens, self.num_outputs))
        b_o  = nn.Parameter(torch.zeros(self.num_outputs, device=device, dtype=dtype), requires_grad=True)
        return nn.ParameterList([W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_ho, b_o])
    
    def init_state(self, batch_size, num_hiddens, device=device):
        return torch.zeros((batch_size, num_hiddens), device=device, dtype=dtype)
        
    def gru(self, inputs, state, params):
        outputs = []
        W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_ho, b_o = params
        H = state
        for X in inputs:
            R = torch.sigmoid( torch.matmul(X, W_xr) + torch.matmul(H, W_hr) + b_r )
            Z = torch.sigmoid( torch.matmul(X, W_xz) + torch.matmul(H, W_hz) + b_z )
            H_candidate = torch.tanh( torch.matmul(X, W_xh) + R * torch.matmul(H, W_hh) + b_h )
            H = Z * H + (1 - Z) * H_candidate
            Y = torch.matmul(H, W_ho) + b_o
            outputs.append(Y)
        return outputs, H


# In[5]:


num_hiddens = 256
gruer = GRU(vocab_size, num_hiddens, vocab_size)
params, state = gruer.get_params(), gruer.init_state(1, num_hiddens)
# print(params,state.shape)

gru = gruer.gru
inputs = d2l.to_onehot(torch.tensor([[char_to_idx['分']]]), n_class = vocab_size)
outputs, state = gru(inputs, state, params)

print(inputs[0].shape, outputs[0].shape, state.shape)


# In[51]:


num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

d2l.train_and_predict_rnn(gru, gruer.get_params, gruer.init_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, True, num_epochs, num_steps, lr,
                          clipping_theta, batch_size, pred_period, pred_len,
                          prefixes)


# In[35]:


#简洁实现

num_hiddens=256
num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

lr = 1e-2 # 注意调整学习率
gru_layer = nn.GRU(input_size=vocab_size, hidden_size=num_hiddens)
model = d2l.RNNModel(gru_layer, vocab_size).to(device)
d2l.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)


# ## LSTM
# 
# $$
# I_t = σ(X_tW_{xi} + H_{t−1}W_{hi} + b_i) \\
# F_t = σ(X_tW_{xf} + H_{t−1}W_{hf} + b_f)\\
# O_t = σ(X_tW_{xo} + H_{t−1}W_{ho} + b_o)\\
# \widetilde{C}_t = tanh(X_tW_{xc} + H_{t−1}W_{hc} + b_c)\\
# C_t = F_t ⊙C_{t−1} + I_t ⊙\widetilde{C}_t\\
# H_t = O_t⊙tanh(C_t)
# $$

# In[29]:


class LSTM():
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_outputs = num_outputs
    
    def get_params(self):
        def _set_W(shape):
            W = torch.tensor(np.random.normal(0, 0.01, size=shape), device = device, dtype = dtype)
            return nn.Parameter(W, requires_grad=True)
            
        def set_param():
            W_x = _set_W(shape = (self.num_inputs, self.num_hiddens))
            W_h = _set_W(shape = (self.num_hiddens, self.num_hiddens))
            b   = nn.Parameter(torch.zeros(self.num_hiddens, device=device, dtype=dtype), requires_grad=True)
            return (W_x, W_h, b)
        
        W_xi, W_hi, b_i = set_param()
        W_xf, W_hf, b_f = set_param()
        W_xo, W_ho, b_o = set_param()
        W_xc, W_hc, b_c = set_param()
        
        W_hq = _set_W(shape = (self.num_hiddens, self.num_outputs) )
        b_q  = nn.Parameter(torch.zeros(self.num_outputs, device=device, dtype=dtype), requires_grad=True)
        return nn.ParameterList([W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q])
    
    def lstm(self, inputs, state, params):
        W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q = params
        H, C = state
        outputs = []
        for X in inputs:
            I = torch.sigmoid( torch.matmul(X, W_xi) + torch.matmul(H, W_hi) + b_i )
            F = torch.sigmoid( torch.matmul(X, W_xf) + torch.matmul(H, W_hf) + b_f )
            O = torch.sigmoid( torch.matmul(X, W_xo) + torch.matmul(H, W_ho) + b_o )
            C_candidate = torch.tanh( torch.matmul(X, W_xc) + torch.matmul(H, W_hc) + b_c )
            C = F * C + I * C_candidate
            H = O * torch.tanh(C)
            Y = torch.matmul(H, W_hq) + b_q
            outputs.append(Y)
        return outputs, (H, C)
    
    def init_state(self, batch_size, num_outputs, device = device):
        H = torch.zeros((batch_size, num_outputs), dtype=dtype, device=device)
        C = torch.zeros((batch_size, num_outputs), dtype=dtype, device=device)
        return (H, C)


# In[31]:


num_hiddens = 256
lstmer = LSTM(vocab_size, num_hiddens, vocab_size)
params, state = lstmer.get_params(), lstmer.init_state(1, num_hiddens)
# print(params,state[0].shape)

lstm = lstmer.lstm
inputs = d2l.to_onehot(torch.tensor([[char_to_idx['分']]]), n_class = vocab_size)
outputs, state = lstm(inputs, state, params)

print(inputs[0].shape, outputs[0].shape, state[0].shape)


# In[33]:


num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

d2l.train_and_predict_rnn(lstm, lstmer.get_params, lstmer.init_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, True, num_epochs, num_steps, lr,
                          clipping_theta, batch_size, pred_period, pred_len,
                          prefixes)


# In[36]:


#简洁实现

num_hiddens=256
num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

lr = 1e-2 # 注意调整学习率
lstm_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens)
model = d2l.RNNModel(lstm_layer, vocab_size)
d2l.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)


# ## Deep RNN
# 
# $$
# \boldsymbol{H}_t^{(1)} = \phi(\boldsymbol{X}_t \boldsymbol{W}_{xh}^{(1)} + \boldsymbol{H}_{t-1}^{(1)} \boldsymbol{W}_{hh}^{(1)} + \boldsymbol{b}_h^{(1)})\\
# \boldsymbol{H}_t^{(\ell)} = \phi(\boldsymbol{H}_t^{(\ell-1)} \boldsymbol{W}_{xh}^{(\ell)} + \boldsymbol{H}_{t-1}^{(\ell)} \boldsymbol{W}_{hh}^{(\ell)} + \boldsymbol{b}_h^{(\ell)})\\
# \boldsymbol{O}_t = \boldsymbol{H}_t^{(L)} \boldsymbol{W}_{hq} + \boldsymbol{b}_q
# $$

# In[37]:


num_hiddens=256
num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

lr = 1e-2 # 注意调整学习率

gru_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens,num_layers=2)
model = d2l.RNNModel(gru_layer, vocab_size).to(device)
d2l.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)


# ## Bidirectional RNN
# 
# 
# $$ 
# \begin{aligned} \overrightarrow{\boldsymbol{H}}_t &= \phi(\boldsymbol{X}_t \boldsymbol{W}_{xh}^{(f)} + \overrightarrow{\boldsymbol{H}}_{t-1} \boldsymbol{W}_{hh}^{(f)} + \boldsymbol{b}_h^{(f)})\\
# \overleftarrow{\boldsymbol{H}}_t &= \phi(\boldsymbol{X}_t \boldsymbol{W}_{xh}^{(b)} + \overleftarrow{\boldsymbol{H}}_{t+1} \boldsymbol{W}_{hh}^{(b)} + \boldsymbol{b}_h^{(b)}) \end{aligned} $$
# $$
# \boldsymbol{H}_t=(\overrightarrow{\boldsymbol{H}}_{t}, \overleftarrow{\boldsymbol{H}}_t)
# $$
# $$
# \boldsymbol{O}_t = \boldsymbol{H}_t \boldsymbol{W}_{hq} + \boldsymbol{b}_q
# $$

# In[ ]:


num_hiddens=128
num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e-2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

lr = 1e-2 # 注意调整学习率

gru_layer = nn.GRU(input_size=vocab_size, hidden_size=num_hiddens,bidirectional=True)
model = d2l.RNNModel(gru_layer, vocab_size).to(device)
d2l.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)


# In[ ]:




