'''import needed package'''
%matplotlib inline
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys,os
root_path = os.path.abspath('../')
sys.path.append(os.path.join(root_path, 'input'))
import d2lzh1981 as d2l
import tqdm

'''手写'''
class MLP():
    def __init__(self,num_neurons=[784,10]):
        self.num_neurons = num_neurons
        self.params = self.set_params()

    def set_params(self):
        params = {}
        layer_index = 1
        num_former_layer_neurons = self.num_neurons[0]
        for num_hiddens in self.num_neurons[1:]:
            W = torch.tensor(np.random.normal(0, 0.1, (num_former_layer_neurons, num_hiddens)), dtype = torch.float32)
            b = torch.zeros(num_hiddens, dtype = torch.float32)
            params[f"W{layer_index}"]=W
            params[f"b{layer_index}"]=b
            num_former_layer_neurons = num_hiddens
            layer_index += 1
        return params
    
    def net(self, X):
        X = X.view(X.shape[0], -1)
        O = None
        if len(self.num_neurons) == 2:
            O = torch.add(torch.matmul(X, self.params['W1']), self.params['b1'])
        else:
            H = relu( torch.add(torch.matmul(X, self.params['W1']), self.params['b1']))
            layer_index = 2
            while layer_index < len(self.num_neurons) - 1:
                H = relu( torch.add( torch.matmul(H, self.params[f'W{layer_index}']), self.params[f'b{layer_index}']))
                layer_index += 1
            O = torch.add( torch.matmul(H, self.params[f'W{layer_index}']), self.params[f'b{layer_index}'] )
        return softmax(O)
              
# mlp = MLP(num_neurons = [784,256,64,10])
# for name, param in mlp.params.items():
#     print(name, param.shape)
# print(mlp.net)

#获取数据集
batch_size = 256
train_loader, test_loader = d2l.load_data_fashion_mnist(batch_size,root=os.path.join(root_path, 'input/FashionMNIST2065'))

#定义模型参数
# num_inputs, num_hiddens, num_outputs = 784, 128, 10
# W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype = torch.float32)
# W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype= torch.float32)
# b1 = torch.zeros(num_hiddens, dtype = torch.float32)
# b2 = torch.zeros(num_outputs, dtype = torch.float32)
# params = {}
# params['W1'], params['b1'], params['W2'], params['b2'] = W1, b1, W2, b2

num_inputs, num_hiddens, num_outputs = 784, [256, 64], 10
mlp = MLP(num_neurons = [num_inputs] + num_hiddens + [num_outputs])
params, mlp_net = mlp.params, mlp.net

for param in params.values():
    param.requires_grad_(requires_grad = True)
    
#定义激活函数Relu
def relu(X):
    return torch.max(X, torch.tensor(0.0, dtype = torch.float32))

#定义网络
def net(X):
    X = X.view(X.shape[0], -1)
    H = relu( torch.add( torch.matmul(X, params['W1']), params['b1'] ) )
    O = torch.add( torch.matmul(H, params['W2']), params['b2'] )
    return softmax(O)
    
#定义损失函数
def cross_entropy(y_hat, y):
    return - torch.log( y_hat.gather(dim=1, index = y.view(-1, 1)))

#定义优化器
def sgd(params, lr, batch_size):
    for param in params.values():
        # print(type(param.data), type(lr), type(param.grad),param.grad.shape, type(batch_size))
        param.data -= lr * param.grad / batch_size

#定义softmax
def softmax(X):
    X_exp = X.exp()
    partion = X_exp.sum(dim=1, keepdim = True)
    return X_exp / partion

# 定义评估测试函数
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).sum().item()
        n += y.shape[0]
    return acc_sum / n

def train_script(net=net):
    epochs = 5
    lr = 0.1
    loss = cross_entropy
    optimizer = sgd
    for epoch in range(epochs):
        train_l_sum, train_acc_sum, n = 0, 0, 0
        for X, y in tqdm.tqdm(train_loader):
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            if params['W1'].grad is not None:
                for param in params.values():
                    param.grad.data.zero_()
            l.backward()
            optimizer(params, lr, batch_size)
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_loader, net)
        print(f'epoch: {epoch}, train loss: {train_l_sum / n},train acc: {train_acc_sum / n}, test acc: {test_acc}')

train_script(net=mlp_net)

'''使用pytorch简洁实现'''
import torch.nn as nn
def create_mlp(num_neurons=[num_inputs,num_outputs]):
    net = nn.Sequential()
    net.add_module('flatten',nn.Flatten())
    net.add_module('linear1',nn.Linear(num_neurons[0], num_neurons[1]))
    num_former_layer_neurons = num_neurons[1]
    for i,num_hiddens in enumerate(num_neurons[2:]):
        net.add_module(f'relu{i+2-1}', nn.ReLU())
        net.add_module(f'linear{i+2}', nn.Linear(num_former_layer_neurons, num_hiddens))
        num_former_layer_neurons = num_hiddens
    return net
# net = create_mlp([784, 256, 64, 10])
# print(net)

#定义模型
# net = nn.Sequential(
#         nn.Flatten(),
#         nn.Linear(num_inputs, num_hiddens),
#         nn.ReLU(),
#         nn.Linear(num_hiddens, num_outputs)
#     )

net = create_mlp([784, 256, 64, 10])

#定义损失函数
loss = nn.CrossEntropyLoss()

#定义优化器
optimizer = torch.optim.SGD(net.parameters(), lr = 0.1)

#初始化模型变量
for param in net.parameters():
    nn.init.normal_(param, mean = 0, std = 0.01)
    
def train_torch():
    epochs = 15
    for epoch in range(epochs):
        train_l_sum, train_acc_sum, n = 0, 0, 0
        for X, y in tqdm.tqdm(train_loader):
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_loader, net)
        print(f'epoch :{epoch}, train loss: {train_l_sum / n}, train acc: {train_acc_sum / n}, test acc: {test_acc}')

train_torch()