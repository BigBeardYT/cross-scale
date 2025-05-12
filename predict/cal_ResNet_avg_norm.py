import math

ResNet_mnist_predict = [99.33, 98.84, 99.21, 99.08, 99.16]
avg_mnist = 99.12
print('ResNet_Mnist_Avg_Pred = {}'.format(round(sum(ResNet_mnist_predict) / len(ResNet_mnist_predict), 2)))

ResNet_KMnist_Pred = [95.96, 96.45, 96.69, 95.83, 96.48]
print('ResNet_KMnist_Avg_Pred = {}'.format(round(sum(ResNet_KMnist_Pred) / len(ResNet_KMnist_Pred), 2)))
avg_kmnist = 96.28
# ResNet_Mnist_Avg_Pred = 99.12
# ResNet_KMnist_Avg_Pred = 96.28
ResNet_CIFAR10_Pred = [85.88, 86.83, 86.47, 86.82, 86.21]
avg_cifar10 = round(sum(ResNet_CIFAR10_Pred) / len(ResNet_CIFAR10_Pred), 2)
print('ResNet_CIFAR10_Avg_Pred = {}'.format(avg_cifar10))

epsilons = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
""" 计算标准差 """
temp = 0.0
sum_mnist = 0.0
for i in range(len(ResNet_mnist_predict)):
    sum_mnist += (ResNet_mnist_predict[i] - avg_mnist) ** 2
sigma_mnist = math.sqrt(sum_mnist / len(ResNet_mnist_predict))
print('MNIST数据集的标准差: {:.2f}'.format(sigma_mnist))

sum_kmnist = 0.0
for i in range(len(ResNet_KMnist_Pred)):
    sum_kmnist += (ResNet_KMnist_Pred[i] - avg_kmnist) ** 2
sigma_kmnist = math.sqrt(sum_kmnist / len(ResNet_KMnist_Pred))
print('KMNIST数据集的标准差: {:.2f}'.format(sigma_kmnist))

sum_cifar10 = 0.0
for i in range(len(ResNet_CIFAR10_Pred)):
    sum_cifar10 += (ResNet_CIFAR10_Pred[i] - avg_cifar10) ** 2
sigma_cifar10 = math.sqrt(sum_cifar10 / len(ResNet_CIFAR10_Pred))
print('CIFAR10数据集的标准差 = {:.2f}'.format(sigma_cifar10))

# ResNet_Mnist_Avg_Pred = 99.12
# ResNet_KMnist_Avg_Pred = 96.28
# ResNet_CIFAR10_Avg_Pred = 86.44
# MNIST数据集的标准差: 0.16
# KMNIST数据集的标准差: 0.33
# CIFAR10数据集的标准差 = 0.36
