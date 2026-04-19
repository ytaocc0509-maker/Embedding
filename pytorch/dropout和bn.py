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

base_dir = './dataset'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

filenames = os.listdir('./dataset')

species = ['cloudy', 'rain', 'shine', 'sunrise']

# 创建train和test目录
if not os.path.exists(train_dir):
    os.mkdir(train_dir)

if not os.path.exists(test_dir):
    os.mkdir(test_dir)

# 分别在train和test目录下创建4种类别的目录
# for train_or_test in ['train', 'test']:
#     for spec in species:
#         path = os.path.join(base_dir, train_or_test, spec)
#         os.mkdir(path)

# 要判断一个图片属于哪个类别.
# 'cloudy' in img

# 要把dataset中的图片全部拷贝到train, test目录下的4个子目录中.
# for i, img in enumerate(filenames):
#     for spec in species:
#         if spec in img:
#             img_path = os.path.join(base_dir, img)
#             if i % 5 == 0:
#                 path = os.path.join(base_dir, 'test', spec, img)
#             else:
#                 path = os.path.join(base_dir, 'train', spec, img)
#             shutil.copy(img_path, path)

# 打印每个类别训练数据和测试数据分别有多少图片
for train_or_test in ['train', 'test']:
    for spec in species:
        print(train_or_test, spec, len(os.listdir(os.path.join(base_dir, train_or_test, spec))))

transform = transforms.Compose([
    # 统一缩放到96 * 96
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    # 正则化
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_ds = torchvision.datasets.ImageFolder(train_dir, transform=transform)

test_ds = torchvision.datasets.ImageFolder(test_dir, transform=transform)

batch_size = 32

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=batch_size)


# imgs, labels = next(iter(train_dl))

# # 显示一张图片
# img = imgs[0]

# img = img + 1
# img = img / 2

# # reshape不可以, 会打乱数据. 
# plt.imshow(img.permute(1, 2, 0))
# plt.show()

# 定义模型 卷积神经网络模型
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)  # 卷积1 16 * 94 * 94
        self.pool = nn.MaxPool2d(2, 2)  # 池化 16 * 47 * 47
        self.conv2 = nn.Conv2d(16, 32, 3)  # 卷积2 32 * 45 * 45  -> pooling -> 32 * 22 * 22
        self.conv3 = nn.Conv2d(32, 64, 3)  # 卷积3 64 * 20 * 20  -> pooling -> 64 * 10 * 10
        self.dropout = nn.Dropout()  # dropout 丢弃部分数据

        # batch , channel, height, width, 64, 
        self.fc1 = nn.Linear(64 * 10 * 10, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 4)

        # 前向传播

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # x.view(-1, 64 * 10 * 10)
        x = nn.Flatten()(x)  # 自动处理批次维度
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = Net()
# 把model拷到gpu上
if torch.cuda.is_available():
    model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam 优化器
loss_fn = nn.CrossEntropyLoss()#交叉熵损失函数


#训练和测试深度学习模型
def fit(epoch, model, train_loader, test_loader):
    
    correct = 0
    total = 0
    running_loss = 0
    
    model.train()
    
    # 遍历训练数据
    for x, y in train_loader:
        # 把数据放到GPU上去. 
        x, y = x.to(device), y.to(device)
        y_pred = model(x) #前向传播
        loss = loss_fn(y_pred, y)
        # 反向传播
        optimizer.zero_grad() # 清空梯度
        loss.backward() # 反向传播
        optimizer.step() # 更新参数
        
        # 计算准确率
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1) # 取概率最大的类别
            correct += (y_pred == y).sum().item() # 计算正确预测的样本数
            total += y.size(0) # 计算总样本数
            running_loss += loss.item() # 累加损失
            
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
