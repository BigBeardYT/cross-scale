import torch.nn as nn
import torch.nn.functional as F
import torch


class RenormGroupNorm(nn.Module):
    """
    传入的通道分组数为num_groups，随着特征提取的进行，num_groups会逐渐减少，这也意味着每个group的通道增大，
    因为self.channels_per_group = self.num_channels // self.num_groups，所以每次处理的feature也越来越多
    这也符合重整化的理论，随着对复杂系统的观察（提取特征），尺度逐渐从低到高
    """

    def __init__(self, num_groups, num_channels, scale=0, eps=1e-5):
        super(RenormGroupNorm, self).__init__()
        # 分解的通道数 32
        self.num_groups = num_groups
        # 总的通道数 64
        self.num_channels = num_channels
        self.scale = scale
        self.eps = eps

        # 每个group包含的通道(特征)数
        assert self.num_channels % self.num_groups == 0
        ## num_channels == 64, num_groups == 32, channels_per_group == 2
        self.channels_per_group = self.num_channels // self.num_groups

        # 对于每个group组，定义一个可以学习的缩放和偏移因子gamma beta
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x):
        # ------------如果scale 大于 0， 则让num_groups变小，感受野变大，尺度增加------------#
        if self.scale > 0:
            self.num_groups = (int)(self.num_groups / 2)
            self.channels_per_group = (int)(self.channels_per_group * 2)
            self.scale += 1

        # 解析x的shape，分别为批次数、通道数以及高宽
        batch_size, num_channels, height, width = x.size()
        ## 当self.numgroups随着粗粒度的提升逐渐变为0时，就不用处理分批次处理特征了
        if self.num_groups != 0:
            x = x.view(batch_size, self.num_groups, self.channels_per_group, height, width)
            # 计算每个群组（group）的均值和方差
            mean = x.mean(dim=[2, 3, 4], keepdim=True)
            var = x.var(dim=[2, 3, 4], keepdim=True)
            # 对每个group进行归一化
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = x.view(batch_size, num_channels, height, width)

            # 偏移和缩放
            x = x * self.gamma + self.beta

        else:
            x = x.view(batch_size, num_channels, height, width)
            x = x * self.gamma + self.beta

        # ------------重新初始化gamma和beta，当通道分组小于2时重新初始化两个参数（防止过拟合），并增大尺度参数scale------------ #
        if self.scale or self.channels_per_group < 2:
            self.gamma.data.fill_(1.0)
            self.beta.data.fill_(0.0)
            self.scale += 1
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bn1 = nn.BatchNorm2d(planes)
        # self.rg1 = RenormGroupNorm(32, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.rg2 = RenormGroupNorm(32, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        # out = self.rg1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.rg2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, in_features, block, layers, num_classes):
        self.inplanes = 64
        in_dim = in_features
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.rg1 = RenormGroupNorm(32, 64)
        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.rg2 = RenormGroupNorm(16, 256)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.rg4 = RenormGroupNorm(8, 512)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:  # 看上面的信息是否需要卷积修改，从而满足相加条件
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.rg1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.rg1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.rg2(x)
        x = self.layer4(x)
        x = self.rg4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        outputs = self.fc(x)
        return outputs


def RgResNet18(in_features, num_classes):
    """Constructs a ResNet-18 model."""
    model = ResNet(block=BasicBlock,
                   layers=[2, 2, 2, 2],
                   in_features=in_features,
                   num_classes=num_classes,
                   )
    return model



# model = RgResNet18(3, 10)
# print(model)
# x = torch.rand(32, 3, 224, 224)
# y = model(x)
# print(y)
