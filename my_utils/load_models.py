from RgNormResNet_4.models.resnet import resnet18
from RgNormResNet_4.models.rgresnet import RgResNet18


def get_model(model_name, in_features=3, num_classes=10):
    """ 传入模型名称，以及分类数 """
    if model_name == 'ResNet':
        return resnet18(in_features=in_features, num_classes=num_classes)
    elif model_name == 'RgResNet':
        return RgResNet18(in_features=in_features, num_classes=num_classes)
    else:
        print("输入的模型有误!!!")
        return None
