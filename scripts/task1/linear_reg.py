import torch
import numpy as np
import matplotlib.pyplot as plt
import random

#假设实际线性模型
true_w = torch.Tensor([2,4])
true_b = torch.Tensor([10])
true_w,true_b

#数据集
num_features = 2
num_samples = 1000
features = torch.randn(num_samples, num_features, dtype=torch.float32)
labels = torch.add( torch.matmul(features, true_w), true_b )
noise = torch.tensor(np.random.normal(0,0.01,size=labels.size()),dtype=torch.float32)
labels = torch.add(labels, noise)

#展示数据
# plt.scatter(features[:, 1].numpy(), labels.numpy(), 1);

batch_size = 32
def data_iter(batch_size,features, labels):
    num_samples = len(features)
    indices = list(range(num_samples))
    random.shuffle(indices)
    for i in range(0, num_samples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_samples)]) # the last time may be not enough for a whole batch
        yield  features.index_select(0, j), labels.index_select(0, j)

# for X,y in data_iter(batch_size,features,labels):
#     print(X.shape, y.shape)
#     break

# 初始化模型参数
W = torch.tensor(np.random.normal(0,0.01,(num_features,1)),dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

#定义模型
def linear_reg(X, W, b):
    # print(X.shape, W.shape)
    return torch.add(torch.matmul(X,W),b)

#定义损失函数
def squared_loss(y_hat, y): 
    return (y_hat - y.view(y_hat.size())) ** 2 / 2
    
#定义优化函数
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size

#训练
num_epochs = 10
lr = 0.01

net = linear_reg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, W, b), y).sum()
        l.backward()
        sgd([W,b], lr, batch_size)
        W.grad.data.zero_()
        b.grad.data.zero_()
    train_loss = loss(net(features, W, b), labels)
    print(f'epoch: {epoch}, training loss: {train_loss.mean().item()}')
print(f'true W: {true_w}, true b: {true_b}')
print(f'W: {W}, b: {b}')


#使用pytorch框架简洁实现
import torch.utils.data as Data
dataset = Data.TensorDataset(features,labels)
data_iter = Data.DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = 2)

#定义模型
from torch import nn
from torch import optim
class LinearReg(nn.Module):
    def __init__(self, num_features):
        super(LinearReg, self).__init__()
        self.linear = nn.Linear(num_features,1)
    def forward(self,X):
        y = self.linear(X)
        return y
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_features, 1))

#定义损失函数
loss = nn.MSELoss()

#定义优化器
optimizer = optim.SGD(net.parameters(), lr=0.03)

#训练

#初始化模型参数
nn.init.normal_(net[0].weight, mean=0, std=0.01)
nn.init.constant_(net[0].bias,val=0)

for epoch in range(num_epochs):
    for X,y in data_iter:
        l = loss(net(X),y.view(-1,1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))
print(f'true W: {true_w}, true b: {true_b}')
print(f'true W: {net[0].weight.data}, true b: {net[0].bias.data}')
