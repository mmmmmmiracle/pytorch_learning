#%%
import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from efficientnet_pytorch import EfficientNet



pretrained_net = models.vgg16(pretrained=True)

# pretrained_net = EfficientNet.from_pretrained('efficientnet-b5')

# 打印最后两层网络
print([*pretrained_net.named_children()][-1:])
# [('avgpool', AdaptiveAvgPool2d(output_size=(1, 1))), 
#  ('fc', Linear(in_features=512, out_features=1000, bias=True))]

# 改变全连接层
pretrained_net.fc = nn.Linear(512, 2)
print(pretrained_net.fc)
# Linear(in_features=512, out_features=2, bias=True)

# %%
'''Fine-tuning'''
output_params = list(map(id, pretrained_net.fc.parameters()))
feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())

lr = 0.01
optimizer = optim.SGD([{'params': feature_params},
                       {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}],
                       lr=lr, weight_decay=0.001)


# %%
'''Fixed'''
# 将模型参数设置为不进行梯度更新
for name, layer in pretrained_net.named_children():
    requires_grad = False
    if name == 'fc':
        requires_grad = True
    for param in layer.parameters():
            param.requires_grad = requires_grad
            
# 查看设置结果
for name, layer in pretrained_net.named_children():
    for param in layer.parameters():
        print(f'{name}\trequires grad = {param.requires_grad}' )
        break
#conv1	requires grad = False
#bn1	requires grad = False
#layer1	requires grad = False
#layer2	requires grad = False
#layer3	requires grad = False
#layer4	requires grad = False
#fc	requires grad = True

lr = 0.01
optimizer = optim.SGD(pretrained_net.fc.parameters(), lr=lr, weight_decay=0.001)

# %%
