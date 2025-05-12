import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from RgNormResNet_4.my_utils import set_random_seeds
from RgNormResNet_4.my_utils.load_models import get_model


# 设置随机数种子
# random_seed = 3407
# set_random_seeds.setup_seed(random_seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def my_train(data_name, model_name, num_classes, train_loader, test_loader,
             batch_size, num_epochs, lr, start, end):
    save_name = data_name + '_' + model_name
    if data_name != 'CIFAR10' and data_name != 'SVHN':
        in_features = 1
    else:
        in_features = 3

    for i in range(start, end):
        print("第{}次训练, 模型:{}, 数据集:{}".format(i, model_name, data_name))
        # 加载模型
        model = get_model(model_name, in_features).to(device)
        # 加载优化器
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        train_acc_lst, valid_acc_lst = [], []
        train_loss_lst, valid_loss_lst = [], []
        best_acc = 0.0
        for epoch in range(num_epochs):

            model.train()

            for batch_idx, (features, targets) in enumerate(train_loader):

                ### PREPARE MINIBATCH
                features, targets = features.to(device), targets.to(device)

                ### FORWARD AND BACK PROP
                outputs = model(features)

                predicts = F.softmax(outputs, dim=1)
                loss = F.cross_entropy(outputs, targets)
                optimizer.zero_grad()

                loss.backward()

                ### UPDATE MODEL PARAMETERS
                optimizer.step()
                if batch_idx % 75 == 0:
                    train_loss_lst.append(loss.item())

                ### LOGGING
                if not batch_idx % 200:
                    print(f'Epoch: {epoch + 1:03d}/{num_epochs:03d} | '
                          f'Batch {batch_idx:03d}/{len(train_loader):03d} |'
                          f' Loss: {loss:.4f}')

            # no need to build the computation graph for backprop when computing accuracy
            model.eval()
            with torch.no_grad():
                # 训练精度、训练损失以及测试的精度和损失
                train_acc, train_loss = compute_accuracy_and_loss(model, train_loader, model_name, device=device)
                valid_acc, valid_loss = compute_accuracy_and_loss(model, test_loader, model_name, device=device)
                train_acc_lst.append(train_acc)
                valid_acc_lst.append(valid_acc)
                # train_loss_lst.append(train_loss)
                valid_loss_lst.append(valid_loss)
                print(f'Epoch: {epoch + 1:03d}/{num_epochs:03d} Train Acc.: {train_acc:.2f}%'
                      f' | Validation Acc.: {valid_acc:.2f}%')

            # 保存模型
            if valid_acc > best_acc:
                print('Saving The Best Model...')
                best_acc = valid_acc
                # 最好模型参数的存储路径
                best_model_params_path = '../savemodel/' + save_name + '_bz' \
                                         + str(batch_size) + '_ep' + str(num_epochs) \
                                         + '_lr' + str(lr) + '_seedNone' + str(i) + '.pth'
                torch.save(model.state_dict(), best_model_params_path)
            print('Saving The Last Model...')
            # 保存最后的模型
            last_model_params_path = '../savemodel/' + save_name + '_bz' \
                                     + str(batch_size) + '_ep' + str(num_epochs) \
                                     + '_lr' + str(lr) + '_seedNone' + str(i) + '_last' + '.pth'
            torch.save(model.state_dict(), last_model_params_path)

            # 动态更改学习率
            # if (epoch + 1) == (int)(num_epochs * 0.75):
            #     for params_group in optimizer.param_groups:
            #         params_group['lr'] *= 0.1
            #         print('更改学习率为{}:'.format(params_group['lr']))
            if (epoch + 1) == (int)(num_epochs * 0.75) or (epoch + 1) == (int)(num_epochs * 0.90):
                for params_group in optimizer.param_groups:
                    params_group['lr'] *= 0.1
                    print('更改学习率为{}:'.format(params_group['lr']))

        plt.plot(train_loss_lst, label='loss')
        plt.legend()
        plt.title(save_name + '_train_loss')
        # 存储图片
        plt.savefig('../saveimage/' + save_name + '_bz' + str(batch_size)
                    + '_ep' + str(num_epochs) + '_lr' + str(lr)
                    + '_seedNone' + str(i) + '_trainloss' + '.png', dpi=600)
        # plt.show()
        plt.close()


# 计算精确度和损失
def compute_accuracy_and_loss(model, data_loader, model_name, device):
    correct_pred, num_examples = 0, 0
    cross_entropy = 0.
    for i, (features, targets) in enumerate(data_loader):
        features, targets = features.to(device), targets.to(device)
        outputs = model(features)

        # predicts = F.softmax(outputs, dim=1)
        cross_entropy += F.cross_entropy(outputs, targets).item()
        _, predicted_labels = torch.max(outputs, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100, cross_entropy / num_examples


