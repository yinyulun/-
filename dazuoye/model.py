import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18ForChestXRay(nn.Module):
    def __init__(self, num_classes=2):  # num_classes=2 表示分类为正常与异常
        super(ResNet18ForChestXRay, self).__init__()
        # 加载预训练的ResNet-18模型
        self.model = models.resnet18(pretrained=True)

        # 修改最后一层 fully connected 层的输出节点数为 num_classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# 用于创建模型的函数
def create_resnet18_model(num_classes=2, pretrained=True):
    """
    创建ResNet18模型
    """
    model = ResNet18ForChestXRay(num_classes=num_classes)
    return model


# 示例：如果你需要创建一个模型并打印它的结构
if __name__ == "__main__":
    model = create_resnet18_model(num_classes=2, pretrained=True)
    print(model)
