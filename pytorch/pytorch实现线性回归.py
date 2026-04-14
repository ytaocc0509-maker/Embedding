import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./dataset/Income1.csv')

# print(data)
# plt.scatter(data.Education, data.Income)
# plt.xlabel("Education")
# plt.ylabel("Income")
# plt.show()

# wx + b
# 分解写法
w = torch.randn(1, requires_grad=True)
print(w)
b = torch.zeros(1, requires_grad=True)
print(b)
learning_rate = 0.001  # 学习率的设置

#
X = torch.from_numpy(data.Education.values.reshape(-1, 1).copy()).float()
Y = torch.from_numpy(data.Income.values.reshape(-1, 1).copy()).float()

# 定义训练过程
for epoch in range(5000):
    for x, y in zip(X, Y):
        y_pred = torch.matmul(x, w) + b
        # 损失函数
        loss = (y - y_pred).pow(2).sum()

        # pytorch对一个变量的对此求导，求导的结果会累加起来
        if w.grad is not None:
            # 充值w的函数
            w.grad.data.zero_()
        if b.grad is not None:
            b.grad.data.zero_()
        # 反向传播。即求w,b的倒数
        loss.backward()

        # 更新w,b
        with torch.no_grad():

            w.data -= w.grad.data * learning_rate
            b.data -= b.grad.data * learning_rate


print(w)
print(b)
plt.scatter(data.Education, data.Income)
plt.plot(X.numpy(),(torch.matmul(X, w) + b).data.numpy(), c="r")
plt.show()