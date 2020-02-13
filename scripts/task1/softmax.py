'''import needed package'''
%matplotlib inline
from IPython import display
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms

import sys,os
root_path = os.path.abspath('../')
sys.path.append(os.path.join(root_path,'input'))
import d2lzh1981 as d2l

'''get dataset'''
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

#定义transform
mytransfrom = transforms.Compose([
        transforms.ToTensor(),
    ])

#获取数据集      
mnist_train = torchvision.datasets.FashionMNIST(root=os.path.join(root_path,'input/FashionMNIST2065'), 
    train=True, download=True, transform=mytransfrom)
mnist_test = torchvision.datasets.FashionMNIST(root=os.path.join(root_path, 'input/FashionMNIST2065'), 
    train=False, download=True, transform=mytransfrom)
print(len(mnist_train),len(mnist_test))

#得到dataloader (数据迭代器iterator)
batch_size = 64
num_workers = 4
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers = num_workers)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size = batch_size, shuffle = False, num_workers = num_workers)

'''手写'''
import numpy as np
import tqdm
num_inputs = 28 * 28
num_outputs = 10

W = torch.tensor(np.random.normal(0,0.01,(num_inputs,num_outputs)),dtype=torch.float32)
b = torch.zeros(num_outputs, dtype=torch.float32)
W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

# 定义softmax
def softmax(X):
    X_exp = X.exp()
    partion = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partion

# 定义softmax模型
def net(X):
    return softmax(torch.add(torch.matmul(X.view(-1, num_inputs), W), b))

# 定义cross entropy 损失函数
def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(dim=1, index=y.view(-1, 1)))

# 定义优化函数
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size

# 定义准确率
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()

# 定义评估测试函数
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).sum().item()
        n += y.shape[0]
    return acc_sum / n

# train
def train_script():
    loss = cross_entropy
    num_epochs = 3
    lr = 0.1
    optimizer = sgd
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in tqdm.tqdm(train_loader):
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            # 清空梯度
            if W.grad is not None:
                for param in [W, b]:
                    param.grad.data.zero_()
            # 计算梯度
            l.backward()
            # 更新参数
            optimizer([W, b], lr, batch_size)

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_loader, net)
        print(f'epoch: {epoch}, train loss: {train_l_sum / n},train acc: {train_acc_sum / n}, test acc: {test_acc}')

train_script()

'''使用pytorch框架简易实现'''
#定义网络模型
from torch import nn
from torch.nn import init

class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet,self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
    def forward(self, X): #(batch_size, 1, 28, 28)
        return self.linear(X.view(X.shape[0], -1))
net = LinearNet(num_inputs, num_outputs)

#定义损失函数
loss = nn.CrossEntropyLoss()

#定义优化器
optimizer = torch.optim.SGD(net.parameters(), lr = 0.1)

#训练
#初始化模型参数
init.normal_(net.linear.weight, mean = 0, std = 0.1)
init.constant_(net.linear.bias, val = 0)

def train_torch():
    num_epochs = 3
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in tqdm.tqdm(train_loader):
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            # 清空梯度
            optimizer.zero_grad()
            # 计算梯度
            l.backward()
            # 更新参数
            optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_loader, net)
        print(f'epoch: {epoch}, train loss: {train_l_sum / n},train acc: {train_acc_sum / n}, test acc: {test_acc}')

train_torch()

'''展示预测结果'''
def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    num_cols = 8
    num_rows = len(images) // num_cols
    _, figs = plt.subplots(num_rows, num_cols, figsize=(12, 2 * num_rows))
    for f, img, lbl in zip(figs.reshape(-1), images[:num_cols * num_rows], labels[:num_cols * num_rows]):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()
    
def test(net, X, y):
    y_hat = net(X)
    y_hat = y_hat.argmax(dim=1)
    y, y_hat = y.numpy(), y_hat.numpy()
    y, y_hat = get_fashion_mnist_labels(y), get_fashion_mnist_labels(y_hat)
    labels = [f'{i}\n{j}' for i,j in zip(y,y_hat)]
    show_fashion_mnist(X, labels)

for X, y in test_loader:
    test(net, X, y)
    break

