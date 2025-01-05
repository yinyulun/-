import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 数据增强和预处理
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),        # 随机裁剪并缩放到224x224
    transforms.RandomHorizontalFlip(),         # 随机水平翻转
    transforms.RandomRotation(30),            # 随机旋转 30度
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), # 随机改变亮度、对比度、饱和度和色调
    transforms.ToTensor(),                     # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化到ImageNet的均值和标准差
])

# 数据加载路径
data_dir = 'E:/school/dazuoye/chest_xray'  # 设置为数据集的根路径

# 加载训练集
train_dir = os.path.join(data_dir, 'train')
train_dataset = datasets.ImageFolder(train_dir, transform=transform)

# 加载验证集
val_dir = os.path.join(data_dir, 'val')
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

# 加载测试集
test_dir = os.path.join(data_dir, 'test')
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 打印数据集大小
print(f"训练集大小: {len(train_loader.dataset)}")
print(f"验证集大小: {len(val_loader.dataset)}")
print(f"测试集大小: {len(test_loader.dataset)}")
