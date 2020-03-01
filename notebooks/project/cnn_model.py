# %%
import torch
import torch.nn as nn
import torch.nn.functional as F 
import frn
from dropblock import DropBlock2D
from shakedrop import ShakeDrop

class LeNet(nn.Module):
    def __init__(self, *, channels, fig_size, num_class):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 6, 5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2),
        )
        ##经过卷积和池化层后的图像大小
        fig_size = (fig_size - 5 + 1 + 4 ) // 1
        fig_size = (fig_size - 2 + 2) // 2
        fig_size = (fig_size - 5 + 1) // 1
        fig_size = (fig_size - 2 + 2) // 2
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * fig_size * fig_size, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, num_class),
        )
    def forward(self, X):
        conv_features = self.conv(X)
        output = self.fc(conv_features)
        return output
    
class AlexNet(nn.Module):
    def __init__(self,*, channels, fig_size, num_class):
        super(AlexNet, self).__init__()
        self.dropout = 0.5
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 96, 11, 4),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )
        ##经过卷积和池化层后的图像大小
        fig_size = (fig_size - 11 + 4) // 4
        fig_size = (fig_size - 3 + 2) // 2
        fig_size = (fig_size - 5 + 1 + 4) // 1
        fig_size = (fig_size - 3 + 2) // 2
        fig_size = (fig_size - 3 + 1 + 2) // 1
        fig_size = (fig_size - 3 + 1 + 2) // 1
        fig_size = (fig_size - 3 + 1 + 2) // 1
        fig_size = (fig_size - 3 + 2) // 2 
        self.fc = nn.Sequential(
            nn.Linear(256 * fig_size * fig_size, 4096),
            nn.ReLU(),
            nn.Dropout(p = self.dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p = self.dropout),
            nn.Linear(4096, num_class),
        )
    
    def forward(self, X):
        conv_features = self.conv(X)
        output = self.fc(conv_features.view(X.shape[0], -1))
        return output

class FAlexNet(nn.Module):
    def __init__(self,*, channels, fig_size, num_class):
        super(FAlexNet, self).__init__()
        self.dropout = 0.5
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 96, 11, 4),
            frn.FilterResponseNorm2d(96, learnable_eps=True),
            # nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, 2),
            frn.FilterResponseNorm2d(256, learnable_eps=True),
            # nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1, 1),
            frn.FilterResponseNorm2d(384, learnable_eps=True),
            # nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            frn.FilterResponseNorm2d(384, learnable_eps=True),
            # nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            frn.FilterResponseNorm2d(256, learnable_eps=True),
            # nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )
        ##经过卷积和池化层后的图像大小
        fig_size = (fig_size - 11 + 4) // 4
        fig_size = (fig_size - 3 + 2) // 2
        fig_size = (fig_size - 5 + 1 + 4) // 1
        fig_size = (fig_size - 3 + 2) // 2
        fig_size = (fig_size - 3 + 1 + 2) // 1
        fig_size = (fig_size - 3 + 1 + 2) // 1
        fig_size = (fig_size - 3 + 1 + 2) // 1
        fig_size = (fig_size - 3 + 2) // 2 
        self.fc = nn.Sequential(
            nn.Linear(256 * fig_size * fig_size, 4096),
            nn.ReLU(),
            nn.Dropout(p = self.dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p = self.dropout),
            nn.Linear(4096, num_class),
        )
    
    def forward(self, X):
        conv_features = self.conv(X)
        output = self.fc(conv_features.view(X.shape[0], -1))
        return output

class VggBlock(nn.Module):
    def __init__(self, conv_arch):
        super(VggBlock, self).__init__()
        num_convs, in_channels, out_channels = conv_arch
        self.conv = nn.Sequential()
        for i in range(num_convs):
            self.conv.add_module(f'conv_{i+1}', nn.Conv2d(in_channels, out_channels, 3, padding=1))
            in_channels = out_channels
        self.conv.add_module('maxpool', nn.MaxPool2d(2, 2))
    
    def forward(self, X):
        return self.conv(X)

class Vgg11(nn.Module):
    def __init__(self, *, channels, fig_size, num_class):
        super(Vgg11, self).__init__()
        self.dropout = 0.5
        self.conv_arch = [(1, channels, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512)]
        self.fc_neuros = 4096

        self.vgg_blocks = nn.Sequential()
        for i, conv_arch in enumerate(self.conv_arch):
            self.vgg_blocks.add_module(f'vbb_block{i+1}', VggBlock(conv_arch))

        fig_size = fig_size // (2 ** len(self.conv_arch))
        fc_features = self.conv_arch[-1][-1] * fig_size * fig_size
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_features, self.fc_neuros),
            nn.ReLU(),
            nn.Dropout(p = self.dropout),
            nn.Linear(self.fc_neuros, self.fc_neuros),
            nn.ReLU(),
            nn.Dropout(p = self.dropout),
            nn.Linear(self.fc_neuros, num_class),
        )
    
    def forward(self, X):
        conv_features = self.vgg_blocks(X)
        output = self.fc(conv_features)
        return output
        
class NinBlock(nn.Module):
    def __init__(self, conv_arch):
        # conv_arch : (in_channels, out_channels, kernel_size, stride, padding)
        super(NinBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(*conv_arch),
            nn.ReLU(),
            nn.Conv2d(conv_arch[1], conv_arch[1], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(conv_arch[1], conv_arch[1], kernel_size=1),
            nn.ReLU(),
        )
    
    def forward(self, X):
        return self.conv(X)

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    
    def forward(self, X):
        return F.avg_pool2d(X, kernel_size = X.size()[2:])

class Nin(nn.Module):
    def __init__(self, *, channels, fig_size, num_class):
        super(Nin, self).__init__()
        self.dropout = 0.5
        self.conv_arch = [(channels, 96, 11, 4, 0), (96, 256, 5, 1, 2), 
                          (256, 384, 3, 1, 1), (384, num_class, 3, 1, 1)]
        self.nin_blocks = nn.Sequential()
        for i, conv_arch in enumerate(self.conv_arch[:-1]):
            self.nin_blocks.add_module(f'nin_block_{i+1}', NinBlock(conv_arch))
            self.nin_blocks.add_module(f'max_pool_{i+1}', nn.MaxPool2d(3, 2))
        self.nin_blocks.add_module('dropout', nn.Dropout(p = self.dropout))
        self.nin_blocks.add_module(f'nin_block_{len(self.conv_arch)}', NinBlock(self.conv_arch[-1]))
        self.global_avg_pool = GlobalAvgPool2d()
        self.flatten = nn.Flatten()
    
    def forward(self, X):
        conv_features = self.nin_blocks(X)
        avg_pool = self.global_avg_pool(conv_features)
        return self.flatten(avg_pool)
    
class Inception(nn.Module):
    def __init__(self, conv_arch):
        super(Inception, self).__init__()
        in_channels, c1, c2, c3, c4 = conv_arch
        self.path_1 = nn.Conv2d(in_channels, c1, kernel_size = 1)
        self.path_2 = nn.Sequential(
            nn.Conv2d(in_channels, c2[0], kernel_size = 1),
            nn.ReLU(),
            nn.Conv2d(c2[0], c2[1], kernel_size = 3, padding = 1),
        )
        self.path_3 = nn.Sequential(
            nn.Conv2d(in_channels, c3[0], kernel_size = 1),
            nn.ReLU(),
            nn.Conv2d(c3[0], c3[1], kernel_size = 5, padding=2),
        )
        self.path_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 3, stride=1, padding=1),
            nn.Conv2d(in_channels, c4, kernel_size=1),
        )

    def forward(self, X):
        p1 = F.relu(self.path_1(X))
        p2 = F.relu(self.path_2(X))
        p3 = F.relu(self.path_3(X))
        p4 = F.relu(self.path_4(X))
        return torch.cat((p1, p2, p3, p4), dim = 1)

class GoogleNet(nn.Module):
    def __init__(self, *, channels, fig_size, num_class):
        super(GoogleNet, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(channels, 64, 7, 2, 3),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.Conv2d(64, 192, 3, padding=1),
            nn.MaxPool2d(3, 2, 1),
        )
        self.b3 = nn.Sequential(
            Inception([192, 64, (96, 128), (16, 32), 32]),
            Inception([256, 128, (128, 192), (32, 96), 64]),
            nn.MaxPool2d(3, 2, 1),
        )
        self.b4 = nn.Sequential(
            Inception([480, 192, (96, 208), (16, 48), 64]),
            Inception([512, 160, (112, 224), (24, 64), 64]),
            Inception([512, 128, (128, 256), (24, 64), 64]),
            Inception([512, 112, (144, 288), (32, 64), 64]),
            Inception([528, 256, (160, 320), (32, 128), 128]),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b5 = nn.Sequential(
            Inception([832, 256, (160, 320), (32, 128), 128]),
            Inception([832, 384, (192, 384), (48, 128), 128]),
            GlobalAvgPool2d(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, num_class),
        )
        self.Inception_blocks = nn.Sequential(self.b3, self.b4, self.b5)

    def forward(self, X):
        conv_features = self.b1(X)
        conv_features = self.b2(conv_features)
        incep_features = self.Inception_blocks(conv_features)
        return self.fc(incep_features)

class BatchNorm(nn.Module):
    def __init__(self, *, num_features, num_dims):
        super(BatchNorm, self).__init__()
        super(BatchNorm, self).__init__()
        if num_dims == 2:
            shape = (1, num_features) #全连接层输出神经元
        else:
            shape = (1, num_features, 1, 1)  #通道数
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成0和1
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 不参与求梯度和迭代的变量，全在内存上初始化成0
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)
        self.momentum = 0.9
    
    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var, Module实例的traning属性默认为true, 调用.eval()后设成false
        Y, self.moving_mean, self.moving_var = self._batch_norm(self.training, 
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=self.momentum)
        return Y

    def _batch_norm(self, is_training, X, gamma, beta, moving_mean, moving_var, eps, momentum):
        if not is_training:
            # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
            X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
        else:
            assert len(X.shape) in (2, 4)
            if len(X.shape) == 2:
                # 使用全连接层的情况，计算特征维上的均值和方差
                mean = X.mean(dim=0)
                var = ((X - mean) ** 2).mean(dim=0)
            else:
                # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。这里我们需要保持
                # X的形状以便后面可以做广播运算
                mean = X.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
                var = ((X - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            # 训练模式下用当前的均值和方差做标准化
            X_hat = (X - mean) / torch.sqrt(var + eps)
            # 更新移动平均的均值和方差
            moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
            moving_var = momentum * moving_var + (1.0 - momentum) * var
        Y = gamma * X_hat + beta  # 拉伸和偏移
        return Y, moving_mean, moving_var

class BLeNet(nn.Module):
    def __init__(self, *, channels, fig_size, num_class):
        super(BLeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 6, 5, padding=2),
            BatchNorm(num_features=6, num_dims = 4),
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            BatchNorm(num_features=16, num_dims = 4),
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2),
        )
        ##经过卷积和池化层后的图像大小
        fig_size = (fig_size - 5 + 1 + 4 ) // 1
        fig_size = (fig_size - 2 + 2) // 2
        fig_size = (fig_size - 5 + 1) // 1
        fig_size = (fig_size - 2 + 2) // 2
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * fig_size * fig_size, 120),
            BatchNorm(num_features=120, num_dims = 2),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            BatchNorm(num_features=84, num_dims = 2),
            nn.Sigmoid(),
            nn.Linear(84, num_class),
        )
    def forward(self, X):
        conv_features = self.conv(X)
        output = self.fc(conv_features)
        return output

class Residual(nn.Module):
    #可以设定输出通道数、是否使用额外的1x1卷积层来修改通道数以及卷积层的步幅。
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
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
        return Y + X

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_rediduals, first_block=False):
        super(ResBlock, self).__init__()
        if first_block:
            assert in_channels == out_channels # 第一个模块的通道数同输入通道数一致
        block = []
        for i in range(num_rediduals):
            block.append(Residual(in_channels, out_channels, use_1x1conv=not first_block, stride=2-int(first_block)))
            in_channels = out_channels
        self.resi_block = nn.Sequential(*block)

    def forward(self, X):
        return self.resi_block(X)

class ResNet(nn.Module):
    def __init__(self, *, channels, fig_size, num_class):
        super(ResNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, 7, 2, 3),
            nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.MaxPool2d(3, 2, 1),
        )
        self.res_block_arch = [(32, 32, 3, True), (32, 64, 4), (64, 128, 6), (128, 256, 3)]
        self.res_blocks = nn.Sequential()
        for i, arch in enumerate(self.res_block_arch):
            self.res_blocks.add_module(f'res_block_{i+1}', ResBlock(*arch))
        self.global_avg_pool = GlobalAvgPool2d()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, num_class),
        )

    def forward(self, X):
        conv_features = self.conv(X)
        res_features = self.res_blocks(conv_features)
        global_avg_pool = self.global_avg_pool(res_features)
        return self.fc(global_avg_pool)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs):
        super(DenseBlock, self).__init__()
        dense_block = []
        for i in range(num_convs):
            in_ch = in_channels + i * out_channels
            dense_block.append(nn.Sequential(
                nn.BatchNorm2d(in_ch),
                nn.ReLU(),
                nn.Conv2d(in_ch, out_channels, 3, padding=1),
            ))
        self.dense_block = nn.ModuleList(dense_block)
        self.out_channels = in_channels + num_convs * out_channels

    def forward(self, X):
        for block in self.dense_block:
            Y = block(X)
            X = torch.cat((X, Y), dim = 1)
        return X

class TransBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransBlock, self).__init__()
        self.trans_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.AvgPool2d(2, 2),
        )
    def forward(self, X):
        return self.trans_block(X)

class DenseNet(nn.Module):
    def __init__(self, *, channels, fig_size, num_class):
        super(DenseNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
        )
        self.dense_blocks = nn.Sequential()
        self.num_convs_list = [6, 12, 24, 16]
        cur_channels, self.growth_rate = 64, 32
        for i, num_conv in enumerate(self.num_convs_list):
            dense_block = DenseBlock(cur_channels, self.growth_rate, num_conv)
            self.dense_blocks.add_module(f'dense_block_{i+1}', dense_block)
            cur_channels = dense_block.out_channels
            if i != len(self.num_convs_list) - 1:
                self.dense_blocks.add_module(f'transition_block_{i+1}', TransBlock(cur_channels, cur_channels // 2))
                cur_channels //= 2
        self.bn = nn.Sequential(nn.BatchNorm2d(cur_channels), nn.ReLU())
        self.global_avg_pool = GlobalAvgPool2d()
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(cur_channels, num_class))

    def forward(self, X):
        conv_features = self.conv(X)
        dense_features = self.dense_blocks(conv_features)
        batch_normed = self.bn(dense_features)
        global_avg_pool = self.global_avg_pool(batch_normed)
        return self.fc(global_avg_pool)


#%%       

fig_size = 224
channels = 3
num_class = 10
X = torch.ones([10,channels, fig_size, fig_size])

# dense = DenseBlock(3, 10, 34)
# out = dense(X)
# print(out.shape)

# dense = DenseNet(channels = channels, fig_size = fig_size, num_class = num_class)
# output = dense(X)
# print(dense)

# nin = Nin(channels = channels, fig_size = fig_size, num_class = num_class)
# output = nin(X)

# vgg11 = Vgg11(channels = channels, fig_size = fig_size, num_class = num_class)
# output = vgg11(X)

# googlenet = GoogleNet(channels = channels, fig_size = fig_size, num_class = num_class)
# output = googlenet(X)

# lenet = BLeNet(fig_size=fig_size, num_class=num_class, channels=channels)
# output = lenet(X)

# alexnet = AlexNet(fig_size=fig_size, num_class=num_class,channels = channels)
# output = alexnet(X)

# resnet = ResNet(fig_size=fig_size, num_class=num_class, channels=channels)
# output = resnet(X)

# print(output.shape)



# %%
