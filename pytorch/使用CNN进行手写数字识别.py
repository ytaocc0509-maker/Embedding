
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

print(f'PyTorch file: {torch.__file__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

import torchvision
from torchvision import datasets, transforms
## pytorch中使用GPU进行训练
# 1. 把模型转移到GPU上.
# 2. 将每一批次的训练数据转移到GPU上. print(device)


# transforms.ToTensor
# 1. 把数据转化为tensor
# 2. 数据的值转化为0到1之间.
# 3. 会把channel放到第一个维度上.

# transforms用来做数据增强, 数据预处理等功能的.
transformation = transforms.Compose([transforms.ToTensor(),])

train_ds = datasets.MNIST('./', train=True, transform=transformation, download=True)
# 测试数据集
test_ds = datasets.MNIST('./', train=False, transform=transformation, download=True)
# 转换成dataloader
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=256)

images, labels = next(iter(train_dl))
# pytorch中图片的表现形式[batch, channel, hight, width]
#images.shape

#print(labels)

# 1.创建模型
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)# in: 64, 1, 28 , 28 -> out: 64, 32, 26, 26
        self.pool = nn.MaxPool2d((2, 2)) # out: 64, 32, 13, 13
        self.conv2 = nn.Conv2d(32, 64, 3)# in: 64, 32, 13, 13 -> out: 64, 64, 11, 11
        # 再加一层池化操作, in: 64, 64, 11, 11  --> out: 64, 64, 5, 5
        self.linear_1 = nn.Linear(64 * 5 * 5, 256)
        self.linear_2 = nn.Linear(256, 10)

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # flatten
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.linear_1(x))
        x = self.linear_2(x)
        return x

model = Model()

# 把model拷到GPU上面去
model.to(device)

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)


def fit(epoch, model, train_loader, test_loader):
    correct = 0
    total = 0
    running_loss = 0

    for x, y in train_loader:
        # 把数据放到GPU上去.
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            correct += (y_pred == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total

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
          'test_accuracy: ', round(test_epoch_acc))
    return epoch_loss, epoch_acc, test_epoch_loss, test_epoch_acc


epochs = 20
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
