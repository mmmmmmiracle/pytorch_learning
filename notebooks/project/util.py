import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms 
import torch.utils.data as data
from PIL import Image
import matplotlib.pyplot as plt 
import time
import tqdm
import numpy as np
# import os,sys
# sys.path.append(os.path.abspath('./'))
import config as cfg
import os,sys
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import  EarlyStoppingCallback,InferCallback,CheckpointCallback,AccuracyCallback
from sklearn.model_selection import train_test_split

def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

def get_mean_std():
    train_img,_ = load_mnist(os.path.join(cfg.data_root, 'FashionMNIST/raw'))
    train_img = train_img / 255.0
    mean = np.mean(train_img)
    std = np.std(train_img)
    return (mean, std)

def get_transforms(resize=None, *, is_train = True):
    trans_list = []
    mean, std = get_mean_std()
    # trans_list.append(transforms.ToPILImage(None))
    if resize:
        trans_list.append(transforms.Resize(size=resize))
    if is_train:
        # crop_size = int( resize * 0.95 )
        # padding = int(resize * 0.05)
        trans_list.append(transforms.RandomCrop(28, padding=2))
        trans_list.append(transforms.RandomHorizontalFlip())
    trans_list.append(transforms.ToTensor())
    trans_list.append(transforms.Normalize((mean,), (std,)))
    # if is_train:
    #     trans_list.append(transforms.RandomErasing())
    return trans_list

class FashionDataset(data.Dataset):
    def __init__(self, imgs, labels, transforms=None):
        self.imgs = imgs
        self.labels = labels
        self.transforms = transforms
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        image = self.imgs[index]
        label = self.labels[index]
        if self.transforms:
            image = self.transforms(image)
        # sample = {'image':image, 'label':label}
        return image, torch.tensor(label,dtype=torch.long)
        # return sample

def get_dataloader(resize = None, batch_size = 1024, num_workers=16, val_size=0.2):
    # trans_list = get_transforms(resize=resize)
    # trans = transforms.Compose(trans_list)
    root = os.path.join(cfg.data_root,'FashionMNIST/raw')
    train_imgs, train_labels = load_mnist(root, 'train')
    test_imgs, test_labels = load_mnist(root, 't10k')
    size = 28
    train_imgs, test_imgs = train_imgs / 255.0, test_imgs / 255.0
    train_imgs = train_imgs.astype(np.float32)
    test_imgs = test_imgs.astype(np.float32)
    print(train_imgs.dtype)
    train_imgs = train_imgs.reshape(-1, size, size) 
    test_imgs  = test_imgs.reshape(-1, size, size)
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(train_imgs, train_labels, test_size=val_size, random_state=42)
    trans = {
        'train_or_val':transforms.Compose(get_transforms(resize=resize ,is_train=True)),
        'test': transforms.Compose(get_transforms(resize=resize, is_train=False))}
    train_dataset = FashionDataset(train_imgs, train_labels, trans['train_or_val'])
    val_dataset   = FashionDataset(val_imgs, val_labels, trans['train_or_val'])
    test_dataset  = FashionDataset(test_imgs, test_labels, trans['test'])
    train_loader  = data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True,num_workers=num_workers)
    val_loader    = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader   = data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)
    return train_loader, val_loader, test_loader

def load_data_fashion_mnist(resize=None, batch_size=1024, num_workers=16):
    """Download the fashion mnist dataset and then load into memory."""
    train_transform = torchvision.transforms.Compose(get_transforms(resize=resize, is_train=True))
    test_transform  = torchvision.transforms.Compose(get_transforms(resize=resize, is_train=False))
    mnist_train = torchvision.datasets.FashionMNIST(root=cfg.data_root, train=True, download=True, transform=train_transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=cfg.data_root, train=False, download=True, transform=test_transform)

    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False,num_workers=num_workers)

    return train_iter, test_iter

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_fashion_mnist(images, labels):
    """Use svg format to display plot in jupyter"""
    from IPython import display
    display.set_matplotlib_formats('svg')
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((224, 224)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)

def evaluate_accuracy(data_iter, net, device=cfg.device):
    """Evaluate accuracy of a model on the given data set."""
    acc_sum,n = torch.tensor([0],dtype=torch.float32,device=device),0
    for X,y in data_iter:
        # If device is the GPU, copy the data to the GPU.
        X,y = X.to(device),y.to(device)
        net.eval()
        with torch.no_grad():
            y = y.long()
            acc_sum += torch.sum((torch.argmax(net(X), dim=1) == y))  #[[0.2 ,0.4 ,0.5 ,0.6 ,0.8] ,[ 0.1,0.2 ,0.4 ,0.3 ,0.1]] => [ 4 , 2 ]
            n += y.shape[0]
    return acc_sum.item()/n

def train_model(net, train_iter, test_iter, batch_size, optimizer, scheduler, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    best_test_acc = 0
    for epoch in range(num_epochs):
        scheduler.step()
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        if test_acc > best_test_acc:
            print('find best! save at model/best.pth')
            best_test_acc = test_acc
            torch.save(net.state_dict(), 'model/best.pth')
            #utils.save_model({
            #    'arch': args.model,
            #    'state_dict': net.state_dict()
            #}, 'saved-models/{}-run-{}.pth.tar'.format(args.model, run))
def train_with_callbacks(net, train_iter, val_iter, test_iter, criterion, optimizer, num_epochs, scheduler, device=cfg.device):
    runer = SupervisedRunner()
    # train and valid
    runer.train(
        model = net.to(cfg.device),
        criterion = criterion,
        optimizer = optimizer,
        scheduler=scheduler,
        loaders = {"train" : train_iter, 'valid': val_iter},
        callbacks=[AccuracyCallback() ,EarlyStoppingCallback(patience=10, min_delta=0.001),],
        logdir = cfg.log_dir,
        num_epochs=num_epochs,
        verbose=True
    )
    #test
    runer.infer(
        model = net,
        loaders = {'infer' : val_iter},
        callbacks=[
            CheckpointCallback(resume=os.path.join(cfg.log_dir, 'checkpoints/best.pth')),
            InferCallback(),
            ]
    )

    acc_sum,n = torch.tensor([0],dtype=torch.float32,device=device),0
    for X, y in val_iter:
        X, y = X.to(cfg.device), y.to(cfg.device)
        runner_out = runer.predict_batch({'features': X})['logits']
        with torch.no_grad():
            y = y.long()
            acc_sum += torch.sum((torch.argmax(runner_out, dim=1) == y))  #[[0.2 ,0.4 ,0.5 ,0.6 ,0.8] ,[ 0.1,0.2 ,0.4 ,0.3 ,0.1]] => [ 4 , 2 ]
            n += y.shape[0]
    val_acc = acc_sum.item()/n
    print('valid acc: ', val_acc)
    acc_sum,n = torch.tensor([0],dtype=torch.float32,device=device),0
    for X, y in test_iter:
        X, y = X.to(cfg.device), y.to(cfg.device)
        runner_out = runer.predict_batch({'features': X})['logits']
        with torch.no_grad():
            y = y.long()
            acc_sum += torch.sum((torch.argmax(runner_out, dim=1) == y))  #[[0.2 ,0.4 ,0.5 ,0.6 ,0.8] ,[ 0.1,0.2 ,0.4 ,0.3 ,0.1]] => [ 4 , 2 ]
            n += y.shape[0]
    test_acc = acc_sum.item()/n
    print('test acc: ', test_acc)
    return val_acc, test_acc

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)


def res_visualize(title=None):
    losses = {'train': [], 'val':[]}
    accuracy = {'train': [], 'val':[]}
    log_src = os.path.join(cfg.log_dir, 'log.txt')
    f = open(log_src)
    while f.readline():
        line = f.readline().strip()
        content = line.split(' | ')
        losses['train'].append(float(content[-1].split('=')[-1]))
        accuracy['train'].append(float(content[-2].split('=')[-1]))
        line = f.readline().strip()
        content = line.split(' | ')
        losses['val'].append(float(content[-1].split('=')[-1]))
        accuracy['val'].append(float(content[-2].split('=')[-1]))
        # print(content)
        # break
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(losses['train'])
    plt.plot(losses['val'])
    plt.legend(('train', 'valid'))
    plt.subplot(122)
    plt.plot(accuracy['train'])
    plt.plot(accuracy['val'])
    plt.legend(('train', 'valid'))
    plt.suptitle(title)
    plt.show()




'''pretrained resnet18, fine tune'''
# net = models.resnet18(pretrained=False)
# checkpoint = torch.load('/home/gongxj/students/.cache/torch/checkpoints/resnet18-5c106cde.pth')
# net.load_state_dict(checkpoint)
# net.fc = nn.Linear(512, 10)
# net.conv1 = nn.Conv2d(1, 64, 7, 2, 3)
# net.conv1.weight.data = checkpoint['conv1.weight'].mean(dim=1, keepdim=True)

# output_params = list(map(id, net.fc.parameters())) + list(map(id, net.conv1.parameters()))
# feature_params = filter(lambda p: id(p) not in output_params, net.parameters())
# optimizer = adabound.AdaBound(
#     [{'params': feature_params},
#      {'params': net.fc.parameters(), 'lr': lr * 10}],
#                        lr=lr, weight_decay=0.001
# )