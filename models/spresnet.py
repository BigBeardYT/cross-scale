import torch.nn as nn
import torch.nn.functional as F

from RgNormResNet_3.kWTA import models


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SparseBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, sparsity=0.5, use_relu=True, sparse_func='reg', bias=False):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.use_relu = use_relu
        self.sparse1 = models.sparse_func_dict[sparse_func](sparsity)
        self.sparse2 = models.sparse_func_dict[sparse_func](sparsity)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        if self.use_relu:
            out = self.relu(out)
        out = self.sparse1(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        if self.use_relu:
            out = self.relu(out)
        out = self.sparse2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, bias=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.relu = nn.ReLU()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class SparseBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, sparsity=0.5, use_relu=True, sparse_func='reg', bias=True):
        super(SparseBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.relu = nn.ReLU()

        self.sparse1 = models.sparse_func_dict[sparse_func](sparsity)
        self.sparse2 = models.sparse_func_dict[sparse_func](sparsity)
        self.sparse3 = models.sparse_func_dict[sparse_func](sparsity)

        self.use_relu = use_relu

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        if self.use_relu:
            out = self.relu(out)
        out = self.sparse1(out)

        out = self.bn2(self.conv2(out))
        if self.use_relu:
            out = self.relu(out)
        out = self.sparse2(out)

        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)

        if self.use_relu:
            out = self.relu(out)
        out = self.sparse3(out)
        return out


class SparseResNet(nn.Module):
    def __init__(self, block, in_channels, num_blocks, sparsities, num_classes=10, use_relu=True, sparse_func='reg',
                 bias=True):
        super(SparseResNet, self).__init__()
        self.in_planes = 64
        self.use_relu = use_relu

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, sparsity=sparsities[0],
                                       sparse_func=sparse_func, bias=bias)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, sparsity=sparsities[1],
                                       sparse_func=sparse_func, bias=bias)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, sparsity=sparsities[2],
                                       sparse_func=sparse_func, bias=bias)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, sparsity=sparsities[3],
                                       sparse_func=sparse_func, bias=bias)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.relu = nn.ReLU()

        self.activation = {}

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.cpu().detach()

        return hook

    def register_layer(self, layer, name):
        layer.register_forward_hook(self.get_activation(name))

    def _make_layer(self, block, planes, num_blocks, stride, sparsity=0.5, sparse_func='reg', bias=True):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_planes, planes, stride, sparsity, self.use_relu, sparse_func=sparse_func, bias=bias))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class SparseResNet_ImageNet(nn.Module):
    def __init__(self, block, num_blocks, sparsities, num_classes=1000, sparse_func='vol', bias=False):
        super(SparseResNet_ImageNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, sparsity=sparsities[0],
                                       sparse_func=sparse_func, bias=bias)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, sparsity=sparsities[1],
                                       sparse_func=sparse_func, bias=bias)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, sparsity=sparsities[2],
                                       sparse_func=sparse_func, bias=bias)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, sparsity=sparsities[3],
                                       sparse_func=sparse_func, bias=bias)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.sp = models.sparse_func_dict[sparse_func](sparsities[0])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.activation = {}

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.cpu().detach()

        return hook

    def register_layer(self, layer, name):
        layer.register_forward_hook(self.get_activation(name))

    def _make_layer(self, block, planes, num_blocks, stride, sparsity=0.5, sparse_func='reg', bias=True):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_planes, planes, stride, sparsity, use_relu=False, sparse_func=sparse_func, bias=bias))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.sp(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, in_channels, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.relu = nn.ReLU()

        self.activation = {}

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.cpu().detach()

        return hook

    def register_layer(self, layer, name):
        layer.register_forward_hook(self.get_activation(name))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, 3, [2, 2, 2, 2])


def MnistResNet18():
    return ResNet(BasicBlock, 1, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


# 训练单通道图像的网络
def SparseResNet18(relu=False, sparsities=[0.5, 0.4, 0.3, 0.2], sparse_func='reg', bias=False):
    return SparseResNet(SparseBasicBlock, 3, [2, 2, 2, 2], sparsities, use_relu=relu, sparse_func=sparse_func,
                        bias=bias)


# 能训练单通道图像的ResNet18
def SparseMnistResNet18(relu=False, sparsities=[0.5, 0.4, 0.3, 0.2], sparse_func='reg', bias=False):
    return SparseResNet(SparseBasicBlock, 1, [2, 2, 2, 2], sparsities, use_relu=relu, sparse_func=sparse_func,
                        bias=bias)


def SparseResNet34(relu=False, sparsities=[0.5, 0.4, 0.3, 0.2], sparse_func='reg', bias=False):
    return SparseResNet(SparseBasicBlock, [3, 4, 6, 3], sparsities, use_relu=relu, sparse_func=sparse_func, bias=bias)


def SparseResNet50(relu=False, sparsities=[0.5, 0.4, 0.3, 0.2], sparse_func='reg', bias=False):
    return SparseResNet(SparseBottleneck, [3, 4, 6, 3], sparsities, use_relu=relu, sparse_func=sparse_func, bias=bias)


def SparseResNet101(relu=False, sparsities=[0.5, 0.4, 0.3, 0.2], sparse_func='reg', bias=False):
    return SparseResNet(SparseBottleneck, [3, 4, 23, 3], sparsities, use_relu=relu, sparse_func=sparse_func, bias=bias)


def SparseResNet152(relu=False, sparsities=[0.5, 0.4, 0.3, 0.2], sparse_func='reg', bias=False):
    return SparseResNet(SparseBottleneck, [3, 8, 36, 3], sparsities, use_relu=relu, sparse_func=sparse_func, bias=bias)


def SparseResNet152_ImageNet(relu=False, sparsities=[0.5, 0.4, 0.3, 0.2], sparse_func='reg', bias=False):
    return SparseResNet_ImageNet(SparseBottleneck, [3, 8, 36, 3], sparsities, sparse_func=sparse_func, bias=bias)


########### End resnet related ##################


