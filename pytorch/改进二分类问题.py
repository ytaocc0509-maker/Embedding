import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./dataset/HR.csv')
# print(data.head())
# print(data.info())
# print((data.part.unique()))

# 对于离散的字符串，有两种处理的方式，1.转化为字符串 2.进行noe-hot编码
print(data.join(pd.get_dummies(data.part)).join(pd.get_dummies(data.salary)))

# 把part和salary删掉
data.drop(columns=["part", "salary"], inplace=True)
data.left.value_counts()

# SMOTE
Y_data = data.left.values.reshape(-1, 1).copy()
Y = torch.from_numpy(Y_data.float())

X_data = data[[c for c in data.columns != "left"]].values

X = torch.from_numpy(X_data).type(torch.FloatTensor)

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

    def forward(self):
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
    print('epoch:', epoch, '   ', 'loss', loss_fn(model(X), Y))



print(((model(X).data.numpy() > 0.5) == Y.numpy()).mean())