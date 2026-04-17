import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 数据读取
data = pd.read_csv("dataset_01/credit-a.csv", header=None)

print(data.head())

# 前15列是特征，最后一列是标记
X = data.iloc[:, :-1]
print(X.shape)

Y = data.iloc[:, -1]

# 把标记变成0和1，方便最后求概率
Y.replace(-1, 0, inplace=True)
print(Y.value_counts())
X = torch.from_numpy(X.values).type(torch.FloatTensor)
Y = torch.from_numpy(Y.values.reshape(-1, 1).copy()).float()

# 回归和分类之间，区别其实不大，回归后面加上一层sigmoid就变成了分类
from torch import nn

model = nn.Sequential(
    nn.Linear(15, 1),
    nn.Sigmoid()
)

# print(model)

# BCE binarty cross entroy 二分类的交叉熵损失
loss_fn = nn.BCELoss()

opt = torch.optim.SGD(model.parameters(), lr=0.0001)

# print(X.shape)
batch_size = 32
steps = 653 // 32

for epoch in range(5000):
    # 每次取32个数据
    for batch in range(steps):
        # 起始索引
        start = batch * batch_size
        # 结束索引
        end = start + batch_size
        # 取数据
        x = X[start: end]
        y = Y[start: end]
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        # 梯度清零
        opt.zero_grad()
        # 反向传播
        loss.backward()
        # 更新
        opt.step()

print(model.state_dict())


# 计算正确率
# 设定阈值
# 现在预测得到的是概率，根据阈值，把概率转化为类别，就可计算准确率
print(((model(X).data.numpy() > 0.5) == Y.numpy()).mean())
