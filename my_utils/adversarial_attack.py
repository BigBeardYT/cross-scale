import sys

sys.path.append("..")
import torch
import torch.nn as nn
from RgNormResNet_3.my_utils.pgd import *
from RgNormResNet_3.my_utils.fgsm import *
from RgNormResNet_4.my_utils.load_models import get_model
import torch.nn.functional as F
# 导入datetime模块
from datetime import datetime

# 获取当前日期和时间
now = datetime.now()

# 输出当前日期和时间
print("当前日期为: {}, 时间: {}".format(now.date(), now.strftime("%H:%M:%S")))

device = 'cuda'
epsilons = [0.01, 8/255, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


def noise_attack(noise_name, data_name, model_name,
                 test_loader, num_classes, lr, attack_used_batch_size,
                 batch_size, num_epochs,
                 start, end):
    if noise_name == 'PGD':
        print('PGD攻击...')
    elif noise_name == 'BIM':
        print('BIM攻击...')
    elif noise_name == 'FGSM':
        print('FGSM攻击...')
    else:
        print('输入的攻击方法有误!!!')
        return

    # 输入通道
    in_features = 3
    if data_name != 'CiFar10' and data_name != 'SVHN' and data_name != 'CIFAR10':
        in_features = 1

    for i in range(start, end):
        # 模型对抗攻击
        print('第{}次攻击, 模型 {}, 数据集 {}'.format(i, model_name, data_name))
        # 加载模型
        attacked_model = get_model(model_name, in_features=in_features, num_classes=num_classes).to(device)
        # print('模型:', attacked_model)
        # 路径
        attacked_model_params_path = '../savemodel/' + data_name + '_' + model_name \
                                     + '_bz' + str(batch_size) + '_ep' + str(num_epochs) + \
                                     '_lr' + str(lr) + '_seedNone' + str(i) + '.pth'
        # attacked_model_params_path = '../trained_model/' + data_name + '_' + model_name + '_' + 'PGD' \
        #                              + '_train' + '_bz' + str(batch_size) + '_ep' + str(num_epochs) + \
        #                              '_lr' + str(lr) + '_seedNone' + str(i) + '.pth'

        # attacked_model_params_path = '../trades_trained_model/CiFar10_ResNet_Trades_train_bz128_ep100_lr0
        # .01_seedNone1.pth'

        print('模型参数所在位置\n' + attacked_model_params_path)

        # 加载参数
        attacked_model.load_state_dict(torch.load(attacked_model_params_path))

        """ 扰动攻击 """
        for epsilon in epsilons:

            criterion = nn.CrossEntropyLoss()
            total_correct = 0
            total_samples = 0
            for images, labels in test_loader:

                images, labels = images.to(device), labels.to(device)
                # 原始模型
                init_outputs = attacked_model(images)
                # 预测结果
                _, pred = torch.max(init_outputs, 1)

                total_samples += init_outputs.shape[0]
                # 生成PGD噪声
                if noise_name == 'PGD':
                    iters = generate_pgd_noise(attacked_model, images, labels, criterion, device,
                                               epsilon=epsilon, num_iter=20, minv=0, maxv=1)
                elif noise_name == 'BIM':
                    iters = generate_bim_noise(attacked_model, images, labels, criterion, device,
                                               epsilon=epsilon, iters=5, minv=0, maxv=1)
                elif noise_name == 'FGSM':
                    iters = generate_fgsm_noise(attacked_model, images, labels, criterion, device,
                                                epsilon=epsilon, minv=0, maxv=1)

                eta, adv_images = iters

                # 攻击后的图片的预测结果
                final_outputs = attacked_model(adv_images)
                _, final_preds = torch.max(final_outputs, 1)

                final_preds_list = final_preds.tolist()
                src_labels = labels.tolist()
                for j in range(len(final_preds_list)):
                    if final_preds_list[j] == src_labels[j]:
                        total_correct += 1

            final_acc = total_correct / float(len(test_loader) * attack_used_batch_size)  # 计算整体准确率
            print("Epsilon: {}\tTest Accuracy = {} / {} = {:.2f}".format(
                epsilon, total_correct, len(test_loader) * attack_used_batch_size, final_acc * 100))

#
# def bim_attack(data_name, model_name,
#                test_loader, num_classes,
#                batch_size, num_epochs,
#                start, end):
#     print("BIM 攻击 ... ")
#
#     # 输入通道
#     in_features = 3
#     if data_name != 'CiFar10':
#         in_features = 1
#
#     for i in range(start, end):
#         # 模型对抗攻击
#         print('第{}次攻击, 模型 {}, 数据集 {}'.format(i, model_name, data_name))
#         # 加载模型
#         attacked_model = get_model(model_name, in_features=in_features, num_classes=num_classes).to(device)
#         # print('模型:', attacked_model)
#         # 路径
#         attacked_model_params_path = '../trained_model/' + data_name + '_' + model_name \
#                                      + '_bz' + str(batch_size) + '_ep' + str(num_epochs) + \
#                                      '_lr0.001_seedNone' + str(i) + '.pth'
#         # attacked_model_params_path = '../trained_model/' + 'The_Best_RgNormAlex_bz32_epochs30.pth'
#         print('模型参数所在位置\n' + attacked_model_params_path)
#         # 加载参数
#         attacked_model.load_state_dict(torch.load(attacked_model_params_path))
#
#         """ 对BIM的测试 """
#         for epsilon in epsilons:
#
#             criterion = nn.CrossEntropyLoss()
#             total_correct = 0
#             total_samples = 0
#             for images, labels in test_loader:
#
#                 images, labels = images.to(device), labels.to(device)
#                 # 原始模型
#                 init_outputs = attacked_model(images)
#                 # 预测结果
#                 _, pred = torch.max(init_outputs, 1)
#
#                 total_samples += init_outputs.shape[0]
#                 iters = generate_bim_noise(attacked_model, images, labels, criterion, device,
#                                            epsilon=epsilon, iters=5, minv=0, maxv=1)
#                 eta, adv_images = iters
#
#                 # 攻击后的图片的预测结果
#                 final_outputs = attacked_model(adv_images)
#                 _, final_preds = torch.max(final_outputs, 1)
#
#                 final_preds_list = final_preds.tolist()
#                 src_labels = labels.tolist()
#                 for j in range(len(final_preds_list)):
#                     if final_preds_list[j] == src_labels[j]:
#                         total_correct += 1
#
#             final_acc = total_correct / float(len(test_loader) * 16)  # 计算整体准确率
#             print("Epsilon: {}\tTest Accuracy = {} / {} = {:.2f}".format(
#                 epsilon, total_correct, len(test_loader) * 16, final_acc * 100))
#
#         """ 对BIM的测试 """
#
#
# def fgsm_pgd_attack(model, device, test_loader, epsilon, t=5, USE_PGD=False):
#     correct = 0
#     adv_examples = []
#     """ PGD 攻击 """
#     if USE_PGD:
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             data.requires_grad = True  # 以便对输入求导 ** 重要 **
#             output = model(data)
#             init_pred = output.max(1, keepdim=True)[1]
#             if init_pred.item() != target.item():  # 如果不扰动也预测不对，则跳过
#                 continue
#
#             alpha = epsilon / t  # 每次只改变一小步
#             perturbed_data = data
#             final_pred = init_pred
#             for i in range(t):  # 共迭代 t 次
#
#                 loss = F.cross_entropy(output, target)
#                 model.zero_grad()
#                 loss.backward(retain_graph=True)
#                 data_grad = data.grad.data  # 输入数据的梯度 ** 重要 **
#
#                 sign_data_grad = data_grad.sign()  # 取符号（正负）
#                 perturbed_image = perturbed_data + alpha * sign_data_grad  # 添加扰动
#                 perturbed_data = torch.clamp(perturbed_image, 0, 1)  # 把各元素压缩到[0,1]之间
#
#                 output = model(perturbed_data)  # 代入扰动后的数据
#                 final_pred = output.max(1, keepdim=True)[1]  # 预测选项
#
#             # 统计准确率并记录，以便后面做图
#             if final_pred.item() == target.item():
#                 correct += 1
#                 if (epsilon == 0) and (len(adv_examples) < 5):
#                     adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
#                     adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
#             else:  # 保存扰动后错误分类的图片
#                 if len(adv_examples) < 5:
#                     adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
#                     adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
#     # FGSM攻击
#     else:
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             data.requires_grad = True  # 以便对输入求导 ** 重要 **
#             output = model(data)
#             init_pred = output.max(1, keepdim=True)[1]
#             if init_pred.item() != target.item():  # 如果不扰动也预测不对，则跳过
#                 continue
#             loss = F.cross_entropy(output, target)
#             model.zero_grad()
#             loss.backward()
#
#             data_grad = data.grad.data  # 输入数据的梯度 ** 重要 **
#             sign_data_grad = data_grad.sign()  # 取符号（正负）
#             perturbed_image = data + epsilon * sign_data_grad  # 添加扰动
#             perturbed_data = torch.clamp(perturbed_image, 0, 1)  # 把各元素压缩到[0,1]之间
#
#             output = model(perturbed_data)  # 代入扰动后的数据
#             final_pred = output.max(1, keepdim=True)[1]
#
#             # 统计准确率并记录，以便后面做图
#             if final_pred.item() == target.item():
#                 correct += 1
#                 if (epsilon == 0) and (len(adv_examples) < 5):
#                     adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
#                     adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
#             else:  # 保存扰动后错误分类的图片
#                 if len(adv_examples) < 5:
#                     adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
#                     adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
#     # 打印最终结果
#     final_acc = correct / float(len(test_loader))  # 计算整体准确率
#     print("Epsilon: {}\tTest Accuracy = {} / {} = {:.4f}".format(
#         epsilon, correct, len(test_loader), final_acc))
#     return final_acc, adv_examples
#
#
# def my_attack(data_name, model_name,
#               test_loader, num_classes,
#               batch_size, num_epochs, USE_PGD,
#               start, end):
#     # 攻击扰动因子
#
#     if USE_PGD:
#         print('PGD攻击 ...')
#     #     epsilons = [0.01, 0.05, 0.1, 0.15, 0.2]
#     else:
#         print('FGSM攻击 ...')
#     #     epsilons = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
#
#     in_features = 1
#     if data_name == 'CiFar10':
#         in_features = 3
#     for i in range(start, end):
#         # 模型对抗攻击
#         Attacked_Model_accuracies = []
#         Attacked_Model_adv_examples = []
#         print('第{}次攻击, 模型 {}, 数据集 {}'.format(i, model_name, data_name))
#         # 加载模型、s
#         attacked_model = get_model(model_name, in_features=in_features, num_classes=num_classes).to(device)
#         # print('模型:', attacked_model)
#         # 路径
#         attacked_model_params_path = '../trained_model/' + data_name + '_' + model_name \
#                                      + '_bz' + str(batch_size) + '_ep' + str(num_epochs) + \
#                                      '_lr0.001_seedNone' + str(i) + '.pth'
#         print('模型参数路径\n' + attacked_model_params_path)
#
#         attacked_model.load_state_dict(torch.load(attacked_model_params_path))
#
#         # 开始攻击
#         for epsilon in epsilons:
#             # FGSM攻击
#             acc, ex = fgsm_pgd_attack(attacked_model, device, test_loader, epsilon, t=5, USE_PGD=USE_PGD)
#             Attacked_Model_accuracies.append(acc)
#             Attacked_Model_adv_examples.append(ex)








            # # 绘制结果
            # plt.plot(Attacked_Model_accuracies, label='adv_acc')
            # plt.legend()
            # # 存储图片
            # plt.savefig('../save_adv_img/' + save_name + '_ep30_bz32_' + str(i) + '.png')
            # plt.show()

#     if USE_PGD:
#         """ PGD对抗攻击 """
#         for datas, targets in test_loader:
#             datas, targets = datas.to(device), targets.to(device)
#             datas.requires_grea = True
#             outputs = model(datas)
#             _, init_preds = torch.max(outputs, 1)
#
#             alpha = epsilon / t  # 每次只改变一小步
#             perturbed_datas = datas
#             for i in range(t):  # 共迭代 t 次
#                 loss = F.cross_entropy(outputs, targets)
#                 model.zero_grad()
#                 loss.backward(retain_graph=True)
#                 data_grads = datas.grad.data  # 输入数据的梯度 ** 重要 **
#
#                 sign_data_grads = data_grads.sign()  # 取符号（正负）
#                 perturbed_images = perturbed_datas + alpha * sign_data_grads  # 添加扰动
#                 perturbed_datas = torch.clamp(perturbed_images, 0, 1)  # 把各元素压缩到[0,1]之间
#
#                 final_outputs = model(perturbed_datas)  # 代入扰动后的数据
#                 _, final_preds = torch.max(final_outputs, 1)
#
#                 final_preds_list = final_preds.tolist()
#                 src_labels = targets.tolist()
#                 for j in range(len(src_labels)):
#                     if final_preds_list[j] == src_labels[j]:
#                         total_correct += 1
#
#
#     else:
#         """ FGSM对抗攻击 """
#         for datas, targets in test_loader:
#             datas, targets = datas.to(device), targets.to(device)
#             datas.requires_grea = True
#             outputs = model(datas)
#             _, init_preds = torch.max(outputs, 1)
#
#             loss = F.cross_entropy(outputs, targets)
#             model.zero_grad()
#             loss.backward()
#
#             data_grads = datas.grad.data  # 输入数据的梯度 ** 重要 **
#             sign_data_grads = data_grads.sign()  # 取符号（正负）
#             perturbed_images = datas + epsilon * sign_data_grads  # 添加扰动
#             perturbed_datas = torch.clamp(perturbed_images, 0, 1)  # 把各元素压缩到[0,1]之间
#
#             final_outputs = model(perturbed_datas)  # 代入扰动后的数据
#             _, final_preds = torch.max(final_outputs, 1)
#             final_preds_list = final_preds.tolist()
#             src_labels = targets.tolist()
#             for j in range(len(src_labels)):
#                 if final_preds_list[j] == src_labels[j]:
#                     total_correct += 1
#
#         final_acc = total_correct / float(len(test_loader) * 16)  # 计算整体准确率
#         print("Epsilon: {}\tTest Accuracy = {} / {} = {:.2f}".format(
#             epsilon, total_correct, len(test_loader) * 16, final_acc * 100))
