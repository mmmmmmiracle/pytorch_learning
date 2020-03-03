import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # TODO:
import sys
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import config as cfg
import tqdm 
from adabound import AdaBound
from shakedrop import ShakeDrop
from dropblock import DropBlock2D
import math
import torchvision.models as models
import albumentations as albu
from cnn_model import *

class GlobalAvgPool2d(nn.Module):
    """
    全局平均池化层
    可通过将普通的平均池化的窗口形状设置成输入的高和宽实现
    """
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


class Residual(nn.Module): 
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        """
            use_1×1conv: 是否使用额外的1x1卷积层来修改通道数
            stride: 卷积层的步幅, resnet使用步长为2的卷积来替代pooling的作用，是个很赞的idea
        """
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    '''
    resnet block
    num_residuals: 当前block包含多少个残差块
    first_block: 是否为第一个block
    一个resnet block由num_residuals个残差块组成
    其中第一个残差块起到了通道数的转换和pooling的作用
    后面的若干残差块就是完成正常的特征提取
    '''
    if first_block:
        assert in_channels == out_channels # 第一个模块的输出通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)


# 定义resnet模型结构
net = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),   # TODO: 缩小感受野, 缩channel
        nn.BatchNorm2d(64),
        nn.ReLU())
        #nn.ReLU(),
        #nn.MaxPool2d(kernel_size=2, stride=2))   # TODO：去掉maxpool缩小感受野

# 然后是连续4个block
net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))   # TODO: channel统一减半
net.add_module("resnet_block2", resnet_block(64, 128, 2))
net.add_module("resnet_block3", resnet_block(128, 256, 2))
net.add_module("resnet_block4", resnet_block(256, 512, 2))
# global average pooling
net.add_module("global_avg_pool", GlobalAvgPool2d()) 
# fc layer
net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(512, 10)))

# net = ResNet(channels=1, fig_size=cfg.fig_size, num_class=10)
# net = DenseNet(channels=1, fig_size=cfg.fig_size, num_class=10)
print('打印网络结构(主要是为了确认如何调整)')
print(net)


# print('打印 1*1*28*28 输入经过每个模块后的shape')
# X = torch.rand((10, 1, 28, 28))
# for name, layer in net.named_children():
#     X = layer(X)
#     print(name, ' output shape:\t', X.shape)

'''extra augmentation'''
import numpy as np
def strong_aug(p=0.5):
    return albu.Compose([
        albu.RandomResizedCrop(cfg.fig_size, cfg.fig_size),
        albu.RandomContrast(),
        albu.RandomBrightness(),
        albu.RandomGridShuffle(),
    ], p=p)


def load_data_fashion_mnist(batch_size, root='/home/gongxj/students/houys/git_repo/pytorch_learning/inputs/fashion-mnist'):
    """Download the fashion mnist dataset and then load into memory."""

    normalize = transforms.Normalize(mean=[0.2860483762389695], std=[0.3529184201347597])
    train_augs = transforms.Compose([
            transforms.Resize(cfg.fig_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(),    # random erase
        ])
    
    test_augs = transforms.Compose([
            transforms.Resize(cfg.fig_size),
            transforms.ToTensor(),
            normalize
        ])
    
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=train_augs)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=test_augs)
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter

print('训练...')
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)



def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    net.eval() 
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            n += y.shape[0]
    net.train() # 改回训练模式
    return acc_sum / n

def train_model(net, train_iter, test_iter, batch_size, optimizer, scheduler, device, num_epochs, tolerance=20, last_acc=0):
    import numpy as np
    net = net.to(device)
    tmp = tolerance
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    best_test_acc = last_acc
    def _accuracy(y_hat, y):
        return  (y_hat.argmax(dim=1) == y).sum().cpu().item()
    for epoch in range(num_epochs):
        if tolerance <= 0:
            print('the best accuarcy is: ', best_test_acc)
            break
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        # for X, y in tqdm.tqdm(train_iter):
        #     X = X.permute(0, 2, 3, 1).numpy()
        #     # print(X.shape)
        #     augmentation = strong_aug()
        #     images = np.zeros_like(X)
        #     for i, image in enumerate(X):
        #         image = {'image': image}
        #         images[i] = augmentation(**image)['image']
        #     X = torch.tensor(images, dtype=torch.float32)
        #     X = X.permute(0,3, 1, 2)
        #     X = X.to(device)
        #     y = y.to(device)
        #     y_hat = net(X)
        #     l = loss(y_hat, y)
        #     optimizer.zero_grad()
        #     l.backward()
        #     optimizer.step()
        #     train_l_sum += l.cpu().item()
        #     train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
        #     n += y.shape[0]
        #     batch_count += 1

        '''Mixup'''
        # for (images, labels) in tqdm.tqdm(train_iter):
        #     mixup_alpha = 0.1
        #     images, labels = images.to(cfg.device), labels.to(cfg.device)
        #     l = np.random.beta(mixup_alpha, mixup_alpha)
        #     index = torch.randperm(images.size(0))
        #     images_a, images_b = images, images[index]
        #     labels_a, labels_b = labels, labels[index]
        #     mixed_images = l * images_a + (1 - l) * images_b
        #     outputs = net(mixed_images)
        #     L = l * loss(outputs, labels_a) + (1 - l) * loss(outputs, labels_b)
        #     optimizer.zero_grad()
        #     L.backward()
        #     optimizer.step()
        #     train_l_sum += L.cpu().item()
        #     train_acc_sum = l * _accuracy(outputs, labels_a) + (1 - l) * _accuracy(outputs, labels_b)
        #     n += labels.shape[0]
        #     batch_count += 1

        '''RICAP'''
        for X, y in tqdm.tqdm(train_iter):
            beta = 0.1
            I_x, I_y = X.size()[2:]
            w = int(np.round(I_x * np.random.beta(beta, beta)))
            h = int(np.round(I_y * np.random.beta(beta, beta)))
            w_ = [w, I_x - w, w, I_x - w]
            h_ = [h, h, I_y - h, I_y - h]
            cropped_images = {}
            c_ = {}
            W_ = {}
            for k in range(4):
                index = torch.randperm(X.size(0))
                x_k = np.random.randint(0, I_x - w_[k] + 1)
                y_k = np.random.randint(0, I_y - h_[k] + 1)
                cropped_images[k] = X[index][:, :, x_k:x_k + w_[k], y_k:y_k + h_[k]]
                c_[k] = y[index].to(device)
                W_[k] = w_[k] * h_[k] / (I_x * I_y)
            patched_images = torch.cat(
                (torch.cat((cropped_images[0], cropped_images[1]), 2),
                torch.cat((cropped_images[2], cropped_images[3]), 2)), 3)
            output = net(patched_images.to(cfg.device))
            l = sum([W_[k] * loss(output, c_[k].to(cfg.device)) for k in range(4)])
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            n += y.shape[0]
            batch_count += 1
            train_acc_sum += sum([W_[k] * _accuracy(output, c_[k]) for k in range(4)])
            # print(output.shape)

        test_acc = evaluate_accuracy(test_iter, net)
        
        if test_acc > best_test_acc:
            tolerance = tmp
            print(f'find best! save at {os.path.join(cfg.model_saved_path, "best.pth")}')
            best_test_acc = test_acc
            torch.save(net.state_dict(), os.path.join(cfg.model_saved_path, "best.pth"))
            #utils.save_model({
            #    'arch': args.model,
            #    'state_dict': net.state_dict()
            #}, 'saved-models/{}-run-{}.pth.tar'.format(args.model, run))
        else:
            tolerance -= 1

        print('epoch %d, loss %.4f, train acc %.4f, test acc %.4f, time %.1f sec, tolerance %2d, learning rate %.4f'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start, tolerance, scheduler.get_lr()[-1]))

        scheduler.step()
    print('the best accuarcy is: ', best_test_acc)
    return best_test_acc

from torch.optim.lr_scheduler import (
    StepLR, CosineAnnealingLR,ReduceLROnPlateau,MultiStepLR,ExponentialLR, CosineAnnealingWarmRestarts
)
from tricks import *
warm_epoch = 10
epoch = 100
warm_lr = 0.01
lr = 0.1 * batch_size / 256
device = cfg.device
'''warm start'''
# optimizer = optim.SGD(net.parameters(),lr=warm_lr,  momentum=0.9, weight_decay=5e-4) 
# # optimizer = optim.Adam(net.parameters(), lr=warm_lr)
# # optimizer = AdaBound(net.parameters())
# scheduler =  GradualWarmupScheduler(optimizer, lr / warm_lr , warm_epoch)
# best_test_acc = train_model(net, train_iter, test_iter, batch_size, optimizer, scheduler, device,warm_epoch )
# '''train'''
# # scheduler = StepLR(optimizer, 10, 0.6)
# optimizer = optim.SGD(net.parameters(),lr=lr,  momentum=0.9, weight_decay=5e-4) 
# # optimizer = optim.Adam(net.parameters(), lr=lr)
# scheduler = CosineAnnealingLR(optimizer, epoch, 1e-5)  
# net.load_state_dict(torch.load(os.path.join(cfg.model_saved_path, "best.pth")))
# best_test_acc = train_model(net, train_iter, test_iter, batch_size, optimizer, scheduler, device, epoch, last_acc=best_test_acc)

# '''warm retrain'''
# decay = 10
# optimizer = optim.SGD(net.parameters(),lr=warm_lr / decay,  momentum=0.9, weight_decay=5e-4) 
# # optimizer = optim.Adam(net.parameters(), lr=warm_lr/10)
# scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = warm_epoch, T_mult= 1)
# net.load_state_dict(torch.load(os.path.join(cfg.model_saved_path, "best.pth")))
# best_test_acc = train_model(net, train_iter, test_iter, batch_size, optimizer, scheduler, cfg.device, warm_epoch, last_acc=best_test_acc )


'''predict'''
print('加载最优模型')
net.load_state_dict(torch.load(os.path.join(cfg.model_saved_path, "best.pth")))
net = net.to(cfg.device)

print('inference测试集')
net.eval() 
id = 0
preds_list = []
with torch.no_grad():
    for X, y in test_iter:
        batch_pred = list(net(X.to(cfg.device)).argmax(dim=1).cpu().numpy())
        for y_pred in batch_pred:
            preds_list.append((id, y_pred))
            id += 1

print('生成提交结果文件')
with open('submission.csv', 'w') as f:
    f.write('ID,Prediction\n')
    for id, pred in preds_list:
        f.write('{},{}\n'.format(id, pred))