import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./dataset/HR.csv')
# print(data.head())
# print(data.info())
# print((data.part.unique()))

# 对于离散的字符串，有两种处理的方式，1.转化为字符串 2.进行noe-hot编码
data = data.join(pd.get_dummies(data.part)).join(pd.get_dummies(data.salary))

# 把part和salary删掉
data.drop(columns=["part", "salary"], inplace=True)
data.left.value_counts()

# SMOTE
Y_data = data.left.values.reshape(-1, 1).copy()
Y = torch.from_numpy(Y_data).float()

X_data = data[[c for c in data.columns if c != "left"]].values.astype(np.float32)

X = torch.from_numpy(X_data)

# pytorch中最常用的一种创建模型的方式
# 子类的写法
from torch import nn


class HRModel(nn.Module):
    def __init__(self):
        # 先调用父类的方法
        super().__init__()
        # 定义网络中会用到的东西
        self.lin_1 = nn.Linear(20, 64)
        self.lin_2 = nn.Linear(64, 64)
        self.lin_3 = nn.Linear(64, 1)
        self.activate = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # 定义反向传播
        x = self.lin_1(input)
        x = self.activate(x)
        x = self.lin_2(x)
        x = self.activate(x)
        x = self.lin_3(x)
        x = self.sigmoid(x)
        return x


lr = 0.001


# 定义获取模型的函数和优化器
def get_model():
    model = HRModel()
    return model, torch.optim.Adam(model.parameters(), lr=lr)


# 定义损失函数
loss_fn = nn.BCELoss()

model, opt = get_model()

batch_size = 64
steps = len(data) // batch_size
epochs = 100

# 训练过程
for epoch in range(epochs):
    for i in range(steps):
        start = i * batch_size
        end = start + batch_size
        x = X[start: end]
        y = Y[start: end]
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    #print('epoch:', epoch, '   ', 'loss', loss_fn(model(X), Y))



#print(((model(X).data.numpy() > 0.5) == Y.numpy()).mean())

# 使用dataset重构

# len(data) data.__len__()
# __getitem__() 对应根据索引取数据. data[0]  = data.__getitem__(0)
#data.__

# pytorch中有一个Dataset类, 可以把任意的具有__len__和__getitem__的对象包装成Dataset对象. 
# Dataset自动取数据
from torch.utils.data import TensorDataset


HRdataset = TensorDataset(X, Y)

# 重写训练过程
for epoch in range(epochs):
    for step in range(steps):
        # 取数据不一样了
        x, y = HRdataset[step * batch_size: step * batch_size + batch_size]
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
print('epoch:', epoch, '     ', 'loss: ', loss_fn(model(X), Y))


#使用DataLoader重构

# dataloader可以自动分批取数据
# dataloader是由dataset创建出来的. 
# 有了dataloader就不需要按切片取数据
from torch.utils.data import DataLoader

HR_ds = TensorDataset(X, Y)
HR_dl = DataLoader(HR_ds, batch_size=batch_size)

# 现在取数据就方便了. 
for x, y in HR_dl:
    print(x, y)


for epoch in range(epochs):
    for x, y in HR_dl:
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
print('epoch:', epoch, '     ', 'loss: ', loss_fn(model(X), Y))


#添加验证

# 需要分割出训练数据和测试数据. 
# 我们刚才是把所有数据作为训练数据.
from sklearn.model_selection import train_test_split

# 切割数据--> 分别创建训练数据和测试数据的dataloader--> 训练过程 --> 校验过程
train_x, test_x, train_y, test_y = train_test_split(X_data, Y_data, random_state=5)

# 转化成tensor
train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
test_x = torch.from_numpy(test_x).type(torch.FloatTensor)

train_y = torch.from_numpy(train_y).type(torch.FloatTensor)
test_y = torch.from_numpy(test_y).type(torch.FloatTensor)

# 变成dataset和dataloader
train_ds = TensorDataset(train_x, train_y)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

test_ds = TensorDataset(test_x, test_y)
test_dl = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=True)

# 定义计算准确率的函数
def accuracy(out, yb):
    return ((out.data.numpy() > 0.5) == yb.numpy()).mean()


# pytorch中有训练模式, 和测试/推理模式. model.train(), model.eval()
# 训练模式和测试模型对一些特殊层会有不同的表现. 比如, dropout, bn等. 
epochs = 1000
model , opt = get_model()

for epoch in range(epochs + 1):
    # 训练的时候, 调到训练模式
    model.train()
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        
        loss.backward()
        opt.zero_grad()
        opt.step()
        
    # 每训练100次输出一次测试结果
    if epoch % 100 == 0:
        model.eval()
        with torch.no_grad():
            # 计算测试损失
            valid_loss = sum([loss_fn(model(x), y) for x,y in test_dl])
            acc_mean = np.mean([accuracy(model(x), y) for x, y in test_dl])
        print(epoch, valid_loss / len(test_dl), acc_mean)






