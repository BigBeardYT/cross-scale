import math

RgNormResNet_mnist_predict = [99.36, 99.27, 99.25, 99.32, 99.33]
print('RgNormResNet_Mnist_Avg_Pred = {}'.format(
    round(sum(RgNormResNet_mnist_predict) / len(RgNormResNet_mnist_predict), 2))
)
avg_mnist = 99.31
# ResNet_Mnist_Avg_Pred = 99.31

RgNormResNet_KMnist_Pred = [96.24, 96.45, 96.52, 96.75, 95.96]
print('RgNormResNet_KMnist_Avg_Pred = {}'.format(
    round(sum(RgNormResNet_KMnist_Pred) / len(RgNormResNet_KMnist_Pred), 2))
)
# RgNormResNet_KMnist_Avg_Pred = 96.38
avg_kmnist = 96.38

RgNormResNet_CIFAR10_Pred = [86.57, 86.27, 87.02, 86.98, 86.66]
avg_cifar10 = round(sum(RgNormResNet_CIFAR10_Pred) / len(RgNormResNet_CIFAR10_Pred), 2)
print('ResNet_CIFAR10_Avg_Pred = {}'.format(avg_cifar10))

""" 计算标准差 """
temp = 0.0
sum_mnist = 0.0
for i in range(len(RgNormResNet_mnist_predict)):
    sum_mnist += (RgNormResNet_mnist_predict[i] - avg_mnist) ** 2
sigma_mnist = math.sqrt(sum_mnist / len(RgNormResNet_mnist_predict))
print('MNIST数据集的标准差: {:.2f}'.format(sigma_mnist))

sum_kmnist = 0.0
for i in range(len(RgNormResNet_KMnist_Pred)):
    sum_kmnist += (RgNormResNet_KMnist_Pred[i] - avg_kmnist) ** 2
sigma_kmnist = math.sqrt(sum_kmnist / len(RgNormResNet_KMnist_Pred))
print('KMNIST数据集的标准差: {:.2f}'.format(sigma_kmnist))

sum_cifar10 = 0.0
for i in range(len(RgNormResNet_CIFAR10_Pred)):
    sum_cifar10 += (RgNormResNet_CIFAR10_Pred[i] - avg_cifar10) ** 2
sigma_cifar10 = math.sqrt(sum_cifar10 / len(RgNormResNet_CIFAR10_Pred))
print('CIFAR10数据集的标准差 = {:.2f}'.format(sigma_cifar10))

# RgNormResNet_Mnist_Avg_Pred = 99.31
# RgNormResNet_KMnist_Avg_Pred = 96.38
# ResNet_CIFAR10_Avg_Pred = 86.7
# MNIST数据集的标准差: 0.04
# KMNIST数据集的标准差: 0.27
# CIFAR10数据集的标准差 = 0.28



