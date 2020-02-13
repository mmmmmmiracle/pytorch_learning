#!/usr/bin/env python
# coding: utf-8

# ## 循环神经网络的构造
# 
# 我们先看循环神经网络的具体构造。假设$\boldsymbol{X}_t \in \mathbb{R}^{n \times d}$是时间步$t$的小批量输入，$\boldsymbol{H}_t  \in \mathbb{R}^{n \times h}$是该时间步的隐藏变量，则：
# 
# 
# $$
# \boldsymbol{H}_t = \phi(\boldsymbol{X}_t \boldsymbol{W}_{xh} + \boldsymbol{H}_{t-1} \boldsymbol{W}_{hh}  + \boldsymbol{b}_h).
# $$
# 
# 
# 其中，$\boldsymbol{W}_{xh} \in \mathbb{R}^{d \times h}$，$\boldsymbol{W}_{hh} \in \mathbb{R}^{h \times h}$，$\boldsymbol{b}_{h} \in \mathbb{R}^{1 \times h}$，$\phi$函数是非线性激活函数。由于引入了$\boldsymbol{H}_{t-1} \boldsymbol{W}_{hh}$，$H_{t}$能够捕捉截至当前时间步的序列的历史信息，就像是神经网络当前时间步的状态或记忆一样。由于$H_{t}$的计算基于$H_{t-1}$，上式的计算是循环的，使用循环计算的网络即循环神经网络（recurrent neural network）。
# 
# 在时间步$t$，输出层的输出为：
# 
# 
# $$
# \boldsymbol{O}_t = \boldsymbol{H}_t \boldsymbol{W}_{hq} + \boldsymbol{b}_q.
# $$
# 
# 
# 其中$\boldsymbol{W}_{hq} \in \mathbb{R}^{h \times q}$，$\boldsymbol{b}_q \in \mathbb{R}^{1 \times q}$。
# 

# In[1]:


import torch
import torch.nn as nn
import time
import math
import numpy as np
import sys
sys.path.append("/home/kesci/input")
import d2l_jay9460 as d2l
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
vocab_size = 1027


# ## 手写实现循环网络

# In[2]:


# one_hot
def onehot(x, num_class):
    res = torch.zeros(x.shape[0], num_class, device = device, dtype = dtype)
    res.scatter_(dim=1, index = x.long().view(-1, 1), value=1)
    return res
    
def to_onehot(X, num_class):
    return [ onehot(X[:, i], num_class) for i in range(X.shape[1]) ]

# onehot(torch.tensor(list(range(10)), dtype=dtype), 10)
to_onehot(torch.arange(10).view(2, 5), 10)


# ### RNN类

# In[9]:


class RNN():
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        self.params = self.get_params(num_inputs, num_hiddens, num_outputs)
    
    # 初始化隐藏状态
    def init_state(self,batch_size, num_hiddens):
        return torch.zeros((batch_size, num_hiddens), device=device)
    
    # 初始化模型参数
    def get_params(self, num_inputs, num_hiddens, num_outputs):
        def set_param(shape):
            param = torch.zeros(shape, dtype = dtype, device = device)
            nn.init.normal_(param, mean = 0, std = 0.01)
            return nn.Parameter(param)
        
        W_xh = set_param((num_inputs, num_hiddens))
        W_hh = set_param((num_hiddens, num_hiddens))
        b_h  = torch.nn.Parameter(torch.zeros(num_hiddens, device=device, dtype = dtype))
        W_ho = set_param((num_hiddens, num_outputs))
        b_o  = torch.nn.Parameter(torch.zeros(num_outputs, device=device, dtype = dtype))
        return W_xh, W_hh, b_h, W_ho, b_o
    
    # 定义模型
    def rnn(self, inputs, state, params):
        W_xh, W_hh, b_h, W_ho, b_o = params
        H = state
        outputs = []
        for X in inputs:
            H = torch.tanh(torch.matmul(X,W_xh) + torch.matmul(H, W_hh) + b_h)
            Y = torch.matmul(H, W_ho) + b_o
            outputs.append(Y)
        return outputs, (H, )
        
    # 梯度裁剪
    def grad_clipping(self, params, theta):
        norm = torch.tensor([0.0], device=device)
        for param in params:
            norm += (param.grad.data ** 2).sum()
        norm = norm.sqrt().item()
        if norm > theta:
            for param in params:
                param.grad.data *= (theta / norm)

num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
# rnner = RNN(num_inputs, num_hiddens, num_outputs)
# params, rnn = rnner.params, rnner.rnn
# X = torch.arange(10).view(2, 5).to(device)
# state = rnner.init_state(X.shape[0], num_hiddens)
# inputs = to_onehot(X, vocab_size)
# output, (state, ) = rnn(inputs, state, params)
# print('num steps:',len(output), ' hidden states shape:', state.shape)
# print(output[0].shape)


# ### 定义预测函数

# In[10]:


def predict_rnn(prefix, num_chars, rnn, params, init_state):
    state = init_state(1, num_hiddens)
    # 利用prefix的信息不断更新state
    Y = None
    for word in prefix:
        X = to_onehot(torch.tensor([[char_to_idx[word]]]), num_class=vocab_size)
        Y, (state, ) = rnn(X, state, params)
    #预测
    prefix += idx_to_char[Y[0].argmax(dim=1).item()]
    # print(prefix)
    while num_chars > 0:
        X = to_onehot(torch.tensor([[char_to_idx[prefix[-1]]]]), num_class=vocab_size)
        Y, (state, ) = rnn(X, state, params)
        prefix += idx_to_char[Y[0].argmax(dim=1).item()]
        num_chars -= 1
    return prefix


# In[11]:


rnner = RNN(num_inputs, num_hiddens, num_outputs)
params, rnn = rnner.params, rnner.rnn
print(predict_rnn('分开', 10, rnn, params, rnner.init_state))

rnner = RNN(num_inputs, num_hiddens, num_outputs)
params, rnn = rnner.params, rnner.rnn
print(predict_rnn('不分开', 10, rnn, params, rnner.init_state))

rnner = RNN(num_inputs, num_hiddens, num_outputs)
params, rnn = rnner.params, rnner.rnn
print(predict_rnn('旁边', 10, rnn, params, rnner.init_state))


# In[24]:


num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
rnner = RNN(num_inputs, num_hiddens, num_outputs)
params, rnn = rnner.params, rnner.rnn
def train_and_predict_rnn(rnner, is_random_iter):
    if is_random_iter:
        data_iter_fn = d2l.data_iter_random
    else:
        data_iter_fn = d2l.data_iter_consecutive
    loss = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        if not is_random_iter:
            state = rnner.init_state(batch_size, num_hiddens)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)
        
        for X, Y in data_iter:
            if is_random_iter:  # 如使用随机采样，在每个小批量更新前初始化隐藏状态
                state = rnner.init_state(batch_size, num_hiddens)
            else:  # 否则需要使用detach函数从计算图分离隐藏状态
                for s in state:
                    s.detach()
            inputs = to_onehot(X, vocab_size)
            outputs, (state, ) = rnn(inputs, state, params)
            outputs = torch.cat(outputs, dim=0)
            y = torch.flatten(Y.T)
            l = loss(outputs, y.long())
            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward(retain_graph=True)
            rnner.grad_clipping(params, clipping_theta)  # 裁剪梯度
            d2l.sgd(params, lr, 1)  # 因为误差已经取过均值，梯度不用再做平均
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(prefix, pred_len, rnn, params, rnner.init_state))

train_and_predict_rnn(rnner, True)

train_and_predict_rnn(rnner, False)


# ## 使用pytorch简洁实现

# ### 定义模型

rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)
num_steps, batch_size = 35, 2

class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1) 
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)

    def forward(self, inputs, state):
        # inputs.shape: (batch_size, num_steps)
        X = to_onehot(inputs, vocab_size)
        X = torch.stack(X)  # X.shape: (num_steps, batch_size, vocab_size)
        hiddens, state = self.rnn(X, state)
        hiddens = hiddens.view(-1, hiddens.shape[-1])  # hiddens.shape: (num_steps * batch_size, hidden_size)
        output = self.dense(hiddens)
        return output, state


# ## 定义预测函数

def predict_rnn_pytorch(prefix, num_chars, model, vocab_size, device, idx_to_char,
                      char_to_idx):
    state = None
    output = [char_to_idx[prefix[0]]]  # output记录prefix加上预测的num_chars个字符
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([output[-1]], device=device).view(1, 1)
        (Y, state) = model(X, state)  # 前向计算不需要传入模型参数
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(Y.argmax(dim=1).item())
    return ''.join([idx_to_char[i] for i in output])
    
model = RNNModel(rnn_layer, vocab_size).to(device)
predict_rnn_pytorch('分开', 10, model, vocab_size, device, idx_to_char, char_to_idx)


def train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = d2l.data_iter_consecutive(corpus_indices, batch_size, num_steps, device) # 相邻采样
        state = None
        for X, Y in data_iter:
            if state is not None:
                # 使用detach函数从计算图分离隐藏状态
                if isinstance (state, tuple): # LSTM, state:(h, c)  
                    state[0].detach_()
                    state[1].detach_()
                else: 
                    state.detach_()
            (output, state) = model(X, state) # output.shape: (num_steps * batch_size, vocab_size)
            y = torch.flatten(Y.T)
            l = loss(output, y.long())
            
            optimizer.zero_grad()
            l.backward()
            rnner.grad_clipping(model.parameters(), clipping_theta)
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn_pytorch(
                    prefix, pred_len, model, vocab_size, device, idx_to_char,
                    char_to_idx))


num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e-3, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                            corpus_indices, idx_to_char, char_to_idx,
                            num_epochs, num_steps, lr, clipping_theta,
                            batch_size, pred_period, pred_len, prefixes)





