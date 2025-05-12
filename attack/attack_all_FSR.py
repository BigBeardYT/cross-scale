import sys

sys.path.append("..")
from RgNormResNet_3.my_utils.load_datasets import load_datasets
# 导包，自定义的攻击函数
from RgNormResNet_3.my_utils.adversarial_attack import noise_attack

# 模型（三通道的图，应该加载不同的模型）
FSR_ResNet_model_name = 'FSRResNet'

""" ######## 以下参数训练之前手动设置 ######### """
attacked_batch_size = 128
attacked_num_epochs = 30
lr = 0.01
data_name = 'CiFar10'
model_name = FSR_ResNet_model_name
num_classes = 10
""" ######## 以上参数训练之前手动设置 ######### """


# 攻击时使用的数据集大小
attack_used_batch_size = 128

# 加载数据集
train_dataset, test_dataset, \
    train_loader, test_loader = \
    load_datasets(batch_size=attack_used_batch_size, data_name=data_name)

# 攻击方法
noise_name = 'PGD'
noise_attack(noise_name, data_name, model_name,
             test_loader, num_classes, lr, attack_used_batch_size,
             attacked_batch_size, attacked_num_epochs,
             2, 3)

