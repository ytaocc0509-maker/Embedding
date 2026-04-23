import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import torchvision
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

base_dir = '../dataset'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# transforms.RandomCrop    # 随机位置的裁剪 , CenterCrop 中间位置裁剪
# transforms.RandomRotation # 随机旋转
# transforms.RandomHorizontalFlip() # 水平翻转
# transforms.RandomVerticalFlip() # 垂直翻转
# transforms.ColorJitter(brightness) # 亮度
# transforms.ColorJitter(contrast) # 对比度
# transforms.ColorJitter(saturation) # 饱和度
# transforms.ColorJitter(hue)
# transforms.RandomGrayscale() # 随机灰度化.

# 数据增强只会加在训练数据上. 增强鲁棒性
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),# 调整图片大小
    transforms.RandomCrop(192),# 随机裁剪
    transforms.RandomHorizontalFlip(),# 随机水平翻转
    transforms.RandomVerticalFlip(),# 随机垂直翻转
    transforms.RandomRotation(0.4), # 随机旋转 旋转角度为 0.4 弧度（约 22.9 度）
    #     transforms.ColorJitter(brightness=0.5), #调整亮度
    #     transforms.ColorJitter(contrast=0.5), #调整对比度
    transforms.ToTensor(), # 转换为张量
    # 正则化
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])# 归一化
])

# 测试数据的预处理定义
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # 正则化
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_ds = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
test_ds = torchvision.datasets.ImageFolder(test_dir, transform=test_transform)

batch_size = 32
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
test_dl = DataLoader(test_ds, batch_size=batch_size)

# 加载预训练好的模型
model = torchvision.models.vgg16(pretrained=True)

for param in model.features.parameters():
    param.requires_grad = False

# 修改原网络中的输出层的结构.
model.classifier[-1].out_features = 4

# 拷到gpu上
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# 学习率衰减
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


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

        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1) # 取概率最大的预测类别
            correct += (y_pred == y).sum().item() # 计算正确预测的样本数
            total += y.size(0) # 计算总样本数
            running_loss += loss.item() # 累加损失
    step_lr_scheduler.step() # 学习率衰减
    epoch_loss = running_loss / len(train_loader.dataset) # 训练集损失
    epoch_acc = correct / total # 训练集准确率

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

# plt.plot(range(1, epochs + 1), train_loss, label='train_loss')
# plt.plot(range(1, epochs + 1), test_loss, label='test_loss')
# plt.legend()
# plt.show()

plt.plot(range(1, epochs+1), train_acc, label='train_acc')
plt.plot(range(1, epochs+1), test_acc, label='test_acc')
plt.legend()
plt.show()
