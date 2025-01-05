import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from model import create_resnet18_model  # 导入 model.py 中定义的模型创建函数

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(256),  # 将图像缩放到256x256
    transforms.CenterCrop(224),  # 中心裁剪为224x224
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化到ImageNet的均值和标准差
])

# 数据加载路径
data_dir = 'E:/school/dazuoye/chest_xray'

# 加载测试集
test_dir = os.path.join(data_dir, 'test')
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 载入训练好的模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_resnet18_model(num_classes=2)  # 2类分类，分别是正常与异常
model.load_state_dict(torch.load('model.pth'))  # 加载训练好的模型权重
model = model.to(device)
model.eval()  # 切换到评估模式

# 测试集评估
def evaluate_model_on_test_set(model, test_loader):
    correct_predictions = 0
    total_predictions = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():  # 不计算梯度
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # 统计正确和总的预测数量
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

            # 保存预测值和真实标签
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 输出每个图像的预测标签和正确标签
            for i in range(len(labels)):
                print(f"Image {total_predictions - len(labels) + i + 1}: Predicted Label = {predicted[i].item()} | True Label = {labels[i].item()}")

    accuracy = 100 * correct_predictions / total_predictions
    print("\n==================== Test Results ====================")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Total Samples: {total_predictions}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Total Predictions: {total_predictions}")
    print("\nTest Completed.")

    return accuracy


# 评估模型并输出准确率
evaluate_model_on_test_set(model, test_loader)
