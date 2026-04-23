import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader

import os

# python中的自带的拷贝工具
import shutil

from torchvision import transforms

base_dir = '../dataset'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# 定义图像预处理的流水线
transform = transforms.Compose([
    # 统一缩放到96 * 96
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    # 对张量进行标准化（正则化）处理
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 文件夹中加载图像数据集，自动为每个子文件夹分配一个类别标签
train_ds = torchvision.datasets.ImageFolder(train_dir, transform=transform)

test_ds = torchvision.datasets.ImageFolder(test_dir, transform=transform)

batch_size = 32  # 定义每次模型训练时处理的样本数量（批大小

# 将 Dataset 对象包装成可迭代的数据加载器，用于批量处理数据。
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=batch_size)



# 添加BN层.
# 定义模型
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)  # 16 * 94 * 94
        self.bn1 = nn.BatchNorm2d(16)  # 批归一化层 2D批归一化，对应16个通道
        self.pool = nn.MaxPool2d(2, 2)  # 16 * 47 * 47

        self.conv2 = nn.Conv2d(16, 32, 3)  # 32 * 45 * 45  -> pooling -> 32 * 22 * 22
        self.bn2 = nn.BatchNorm2d(32)  # 批归一化层 2D批归一化，对应32个通道
        self.conv3 = nn.Conv2d(32, 64, 3)  # 64 * 20 * 20  -> pooling -> 64 * 10 * 10
        self.bn3 = nn.BatchNorm2d(64)  # 批归一化层 2D批归一化，对应64个通道
        self.dropout = nn.Dropout()

        # batch , channel, height, width, 64,
        self.fc1 = nn.Linear(64 * 10 * 10, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)  # 批归一化层 1D批归一化，对应全连接1024个通道
        self.dropout = nn.Dropout()
        # batch , channel, height, width, 64,
        self.fc2 = nn.Linear(1024, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)  # 批归一化层 1D批归一化，对应全连接256个通道
        self.fc3 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.bn1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.bn2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.bn3(x)
        # x.view(-1, 64 * 10 * 10)
        x = nn.Flatten()(x)
        x = F.relu(self.fc1(x))
        x = self.bn_fc1(x)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.bn_fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = Net()
# 把model拷到gpu上
if torch.cuda.is_available():
    model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)  # 创建 Adam 优化器 (获取模型的所有训练参数，学习率)
loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数


# 训练和测试深度学习模型
def fit(epoch, model, train_loader, test_loader):
    correct = 0
    total = 0
    running_loss = 0

    model.train()

    # 遍历训练数据
    for x, y in train_loader:
        # 把数据放到GPU上去.
        x, y = x.to(device), y.to(device)
        y_pred = model(x)  # 前向传播
        loss = loss_fn(y_pred, y)
        # 反向传播
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # 计算准确率
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)  # 取概率最大的类别
            correct += (y_pred == y).sum().item()  # 计算正确预测的样本数
            total += y.size(0)  # 计算总样本数
            running_loss += loss.item()  # 累加损失

    # 计算平均损失和准确率
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total

    model.eval()

    # 测试过程
    test_correct = 0
    test_total = 0
    test_running_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()
    test_epoch_loss = test_running_loss / len(test_loader.dataset)
    test_epoch_acc = test_correct / test_total

    print('epoch: ', epoch,
          'loss: ', round(epoch_loss, 3),
          'accuracy: ', round(epoch_acc, 3),
          'test_loss: ', round(test_epoch_loss, 3),
          'test_accuracy: ', round(test_epoch_acc, 3))
    return epoch_loss, epoch_acc, test_epoch_loss, test_epoch_acc





# 模型保存
#torch.save(model.state_dict(), './my_model.pth')

# 恢复模型
# new_model = Net()
# new_model.load_state_dict(torch.load('./my_model.pth'))

#print(model.state_dict())



# 把新的模型拷到GPU上, 进行测试
# new_model.to(device)

# test_correct = 0
# test_total = 0
# new_model.eval()
# with torch.no_grad():
#     for x, y in test_dl:
#         x, y = x.to(device), y.to(device)
#         y_pred = new_model(x)
#         y_pred = torch.argmax(y_pred, dim=1)
#         test_correct += (y_pred == y).sum().item()
#         test_total += y.size(0)

# epoch_test_acc = test_correct / test_total
# print(epoch_test_acc)


# 保存最优参数

import copy

# 没有训练之前的初始化参数, 
best_model_weight = model.state_dict()

epochs = 20
best_acc = 0.0

train_loss = []
train_acc = []
test_loss = []
test_acc = []

for epoch in range(epochs):
    epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch, model, train_dl, test_dl)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)
    
    if epoch_test_acc > best_acc:
        best_acc = epoch_test_acc
        # 更新参数
        best_model_weight = copy.deepcopy(model.state_dict())


# 保存完成模型

# 把最好的参数加载到模型中
model.load_state_dict(best_model_weight)

torch.save(model, 'my_whole_model.pth')

new_model2 = torch.load('my_whole_model.pth')

new_model2.state_dict()

