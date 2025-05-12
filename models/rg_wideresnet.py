import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RenormGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, scale=0, eps=1e-5):
        super(RenormGroupNorm, self).__init__()
        # 分解的通道数
        self.num_groups = num_groups
        # 总的通道数
        self.num_channels = num_channels
        self.scale = scale
        self.eps = eps

        # 每个group包含的通道(特征)数
        assert self.num_channels % self.num_groups == 0
        self.channels_per_group = self.num_channels // self.num_groups

        # 对于每个group组，定义一个可以学习的缩放和偏移因子gamma beta
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x):
        # ------------如果scale 大于 0，则合并相邻的group------------#
        if self.scale > 0:
            self.num_groups *= 2
            self.channels_per_group *= 2
            self.scale += 1
            # self.num_groups += 1
            # self.channels_per_group = self.num_channels // self.num_groups
            # self.scale += 1

        # 解析x的shape，分别为批次数、通道数以及高宽
        batch_size, num_channels, height, width = x.size()
        x = x.view(batch_size, self.num_groups, self.channels_per_group, height, width)
        # 计算每个群组（group）的均值和方差
        mean = x.mean(dim=[2, 3, 4], keepdim=True)
        var = x.var(dim=[2, 3, 4], keepdim=True)

        # ------------重新初始化gamma和beta------------ #
        if self.scale % self.num_groups == 0:
            self.gamma.data.fill_(1.0)
            self.beta.data.fill_(0.0)

        # 对每个group进行归一化
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x.view(batch_size, num_channels, height, width)

        # 偏移和缩放
        x = x * self.gamma + self.beta
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)

        self.rg1 = RenormGroupNorm(out_planes // 2, out_planes)

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.rg2 = RenormGroupNorm(out_planes // 2, out_planes)

        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))

        out = self.rg1(out)

        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        out = self.rg2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class RgWideResNet(nn.Module):
    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0):
        super(RgWideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.rg3 = RenormGroupNorm(nChannels[0] // 4, nChannels[0])
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.rg3(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


# x = torch.rand(1 ,3, 32, 32)
# model = RgWideResNet()
# y = model(x)
# print(y)
# print(WideResNet())

