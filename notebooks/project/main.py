#%%
from cnn_model import *
import adabound 
from util import *
from torch.optim.lr_scheduler import (
    StepLR, CosineAnnealingLR,ReduceLROnPlateau,MultiStepLR,ExponentialLR
)
from torch.optim import SGD, Adam, RMSprop
from ghost_net import ghost_net
from torchvision import models
# import baseline as base

lr , batch_size, num_epochs =1e-2, 512, 100
fig_size = 28
net_name = 'ResNet'
net = eval(f'{net_name}(channels=1, fig_size=fig_size, num_class=10)')
criterion = nn.CrossEntropyLoss()
optimizer = adabound.AdaBound(net.parameters(), lr=lr)
# optimizer = Adam(net.parameters(), lr=lr)
# optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
# scheduler = CosineAnnealingLR(optimizer, 3, 0, last_epoch=-1)
scheduler = StepLR(optimizer, 10, 0.5)
# train_loader, val_loader, test_loader = get_dataloader(resize=fig_size, batch_size=batch_size, num_workers=24,val_size=0.1)
train_loader, test_loader = load_data_fashion_mnist(resize=(fig_size,fig_size), batch_size=batch_size, num_workers=24)
# val_acc, test_acc = train_with_callbacks(net, train_loader, val_loader, test_loader ,criterion, optimizer, num_epochs, scheduler)
# print(checkpoint['conv1.weight'].shape)
# print(net.conv1.weight.data.shape, net.conv1.bias.data.shape)
train_model(net, train_loader, test_loader, batch_size, optimizer, scheduler, cfg.device, num_epochs)
# %%
# val_acc, test_acc = 0.9308333333333333, 0.9319
title = f'result( lr={lr}(cos), optimizer=Adabound)\n val acc: {round(val_acc, 4)}, test acc: {round(test_acc, 4)}'
res_visualize(title)
# %%


