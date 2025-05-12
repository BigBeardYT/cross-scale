import sys

sys.path.append("..")
from RgNormResNet_4.my_utils.load_datasets import load_datasets
# 导包，自定义的攻击函数
from RgNormResNet_4.my_utils.adversarial_attack import noise_attack

# Pretrained_RgNormResNet_model_name = 'Pretrained_RgResNet'

""" ######## 以下参数训练之前手动设置 ######### """
attacked_batch_size = 128
attacked_num_epochs = 100
lr = 0.01
data_name = 'CIFAR10'
model_name = 'RgResNet'
num_classes = 10
""" ######## 以上参数训练之前手动设置 ######### """


# 攻击时使用的数据集大小
attack_used_batch_size = 128

# 加载数据集
train_dataset, test_dataset, \
    train_loader, test_loader = \
    load_datasets(batch_size=attack_used_batch_size, data_name=data_name)

# 攻击方法
noise_name = 'FGSM'
noise_attack(noise_name, data_name, model_name,
             test_loader, num_classes, lr, attack_used_batch_size,
             attacked_batch_size, attacked_num_epochs,
             1, 2)

