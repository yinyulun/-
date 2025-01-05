import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import create_resnet18_model  # 导入 model.py 中定义的模型创建函数
from tqdm import tqdm  # 导入 tqdm 进度条库
from torch.optim import lr_scheduler  # 学习率调度器
import matplotlib.pyplot as plt  # 导入绘图库
from dataset import train_loader, val_loader, test_loader  # 导入 dataset.py 中的数据集加载器

# 2. 实例化模型
model = create_resnet18_model(num_classes=2)  # 2类分类，分别是正常与异常

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数

# 使用Adam优化器，并加入L2正则化（weight_decay）
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2正则化

# 4. 学习率调度器（每7个epoch将学习率减半）
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

# 5. 模型训练和验证
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 用于记录损失和准确率的列表
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# 6. 早停（Early Stopping）相关变量
patience = 5  # 设置在验证集准确率不提升的epoch数为5
best_val_accuracy = 0.0
epochs_without_improvement = 0

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    global epochs_without_improvement, best_val_accuracy

    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        # 使用 tqdm 显示进度条
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)

                # 前向传播
                outputs = model(images)
                loss = criterion(outputs, labels)

                # 反向传播和优化
                optimizer.zero_grad()  # 清除以前的梯度
                loss.backward()  # 计算梯度
                optimizer.step()  # 更新权重

                # 统计训练损失和准确度
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)

                # 更新进度条的显示
                pbar.set_postfix(loss=running_loss / (pbar.n + 1), accuracy=100 * correct_predictions / total_predictions)

        # 计算每个epoch的损失和准确率
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct_predictions / total_predictions
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

        # 在验证集上评估模型
        val_loss, val_accuracy = evaluate_model(model, val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # 早停机制：如果验证准确率没有提升，提前停止训练
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0  # 重置计数器
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping: No improvement in validation accuracy for {patience} epochs")
                break

        # 更新学习率
        scheduler.step()

    # 训练完成后绘制损失和准确率曲线
    plot_metrics()

    # 保存模型
    save_model(model)

def evaluate_model(model, val_loader):
    model.eval()  # 设置模型为评估模式
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():  # 不计算梯度
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

    accuracy = 100 * correct_predictions / total_predictions
    avg_loss = running_loss / len(val_loader)
    print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy

def plot_metrics():
    # 绘制损失和准确率曲线
    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.legend()

    # 显示图表
    plt.tight_layout()
    plt.show()

def save_model(model, model_path='model.pth'):
    # 保存模型的状态字典
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

# 训练模型
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20)

