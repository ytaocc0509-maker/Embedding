import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms

from torch.utils.data import DataLoader

base_dir = '../dataset'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# 定义图像预处理
transform = transforms.Compose([
    # 统一缩放到96 * 96
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    # 正则化
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 加载图像数据集，自动为每个子文件夹分配一个类别标签
train_ds = torchvision.datasets.ImageFolder(train_dir, transform=transform)
test_ds = torchvision.datasets.ImageFolder(test_dir, transform=transform)

batch_size = 32  # 训练批次大小

# 将 Dataset 对象包装成可迭代的数据加载器，用于批量处理数据。
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)

# 加载预训练好的模型
model = torchvision.models.vgg16(pretrained=True)

# 冻结模型预训练的特征提取层，保留其学习到的特征提取能力(迁移学习的核心)
for param in model.features.parameters():
    param.requires_grad = False

# 修改原网络中的输出层的结构.
model.classifier[-1].out_features = 4

# 另一种修改输出层的写法.
# model.classifier[-1] = torch.nn.Linear(model.classifier[-1], 4)

# 拷到gpu上
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)  # 创建 Adam 优化器 (获取模型的所有训练参数，学习率)
loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数


def fit(epoch, model, train_loader, test_loader):
    correct = 0
    total = 0
    running_loss = 0

    # 因为bn和dropout需要手动指定训练模式和推理模式
    model.train()
    for x, y in train_loader:
        # 把数据放到GPU上去.
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():  # 不计算梯度
            # 计算准确率
            y_pred = torch.argmax(y_pred, dim=1)
            correct += (y_pred == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader.dataset)  # 训练损失
    epoch_acc = correct / total  # 测试损失

    # 测试过程
    test_correct = 0
    test_total = 0
    test_running_loss = 0
    model.eval()
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


epochs = 10
train_loss = []
train_acc = []
test_loss = []
test_acc = []
for epoch in range(epochs):
    epoch_loss, epoch_acc, test_epoch_loss, test_epoch_acc = fit(epoch, model, train_dl, test_dl)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)

    test_loss.append(test_epoch_loss)
    test_acc.append(test_epoch_acc)

# plt.plot(range(1, epochs+1), train_loss, label='train_loss')
# plt.plot(range(1, epochs+1), test_loss, label='test_loss')
# plt.legend()
# plt.show()

# plt.plot(range(1, epochs+1), train_acc, label='train_acc')
# plt.plot(range(1, epochs+1), test_acc, label='test_acc')
# plt.legend()
# plt.show()

# 模型保存
# state_dict, 是一个字典, 保存了训练模型.
# print(model.state_dict())

# 保存参数
# path = './vgg16.pth'
# torch.save(model.state_dict(), path)
