import sys
import torch
from torchvision import datasets
from torchvision import transforms
import torch.nn.functional as F
import torch.nn
from RgNormResNet_4.my_utils.load_datasets import load_datasets
from RgNormResNet_4.models.resnet import resnet18
from RgNormResNet_4.my_utils.train_implement import my_train

device = 'cuda' if torch.cuda.is_available() else 'cpu'



# 模型（三通道的图，应该加载不同的模型）

""" ######## 以下参数训练之前手动设置 ######### """
batch_size = 32
data_name = "KMNIST"
num_epochs = 5
lr = 0.01
model_name = "ResNet"

""" ######## 以上参数训练之前手动设置 ######### """


# 加载数据集
train_dataset, test_dataset, \
train_loader, test_loader = \
    load_datasets(batch_size=batch_size, data_name=data_name)

# 不同的数据集不同的分类，这五个数据集中，只有EMNIST是37分类，其他都为10分类
num_classes = 10

my_train(data_name, model_name, num_classes,
         train_loader, test_loader,
         batch_size, num_epochs, lr,
         1, 4)


