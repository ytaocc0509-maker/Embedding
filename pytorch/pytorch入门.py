import torch
import numpy as np

# 打印cuda设备 显卡名字
# print(torch.cuda.get_device_name(0))


# 1 pytorch张量
# pytorch中的张量和tensorflow的tensor是一样，名字都一样，
# pvtorch中的张量也叫tensor
# tensor和numpy中的ndarray也是一个意思，只不过tensor可以在GPU上加速计算


# 创建tensor
print(torch.tensor([1, 2], dtype=torch.int32))  # tensor([1, 2], dtype=torch.int32)
print(torch.tensor((1, 2)))  # tensor([1, 2])
print(torch.tensor(np.array([6, 2])))  # tensor([6, 2], dtype=torch.int32)

# 均匀分布
print(torch.rand(2, 3))  # tensor([[0.6883, 0.8612, 0.3034],[0.0607, 0.6868, 0.8172]])
print(torch.rand((2, 3)))  # tensor([[0.5588, 0.8363, 0.1771],[0.6807, 0.0746, 0.8941]])

# 标准正态分布
print(torch.randn(2, 3))  # tensor([[-0.2305, -0.6192,  1.1476],[ 0.6941,  0.1071, -0.3361]])

# 全零矩阵
print(torch.zeros((2, 3)))  # tensor([[0., 0., 0.],[0., 0., 0.]])

# 全一矩阵
print(torch.ones(2, 3))  # tensor([[1., 1., 1.],[1., 1., 1.]])

# tensor的shape
x = torch.ones((2, 3, 4))
# 可以通过size()方法获取形状
print(x.shape)  # torch.Size([2, 3, 4])
# size中可以传shape的索引
print(x.size(1))  # 3

# numpy和sensor直接的转化
n = np.random.randn(2, 3)
torch.from_numpy((n))

a = torch.from_numpy((n))
a.numpy()

# 张量的和单个数字运算
t = torch.ones(2, 3)
print(t + 3)
print(torch.add(t, 3))

# 两个tensor运算
# 形状要相同
# 对应位置的元素相加,element-wise操作,形状不同的张量运算时会自动触发广播机制，如2×3张量可与标量运算
x1 = torch.ones(2, 3)
print(x1 + t)

# 有输出不会改变原始值
print(t.add(x1))

# t.add_，pytorch中带下划线的操作会改变原始值
print(t.add_(x1))

# 改变tensor的形状
print(t.reshape(3, 2))
print(t.view(3, 2))

# 聚合操作
print(t.mean())  # 取平均
print(t.sum())  # 求和
print(t.sum(dim=1))  # 指定维度进行聚合,不写维度，默认把所有维度聚合

# 去除tensor中的标量
# 一个数字叫做scalars(标量)，带中括号的数据较做向量
# item是专门用来取出tensor中的标量的值
x = x.sum()
print(x.item())

x = torch.rand(3, 4)
print(x[0])
print(x[0, 0])

# tensor的切片和索引操作和ndarray是完全一样
print(x[0, :3])

# 取出图片数据
x = torch.rand(32, 224, 224, 3)
print(x[0, :, :, 0].shape) # torch.Size([224,224])

# 切片取数据
x = torch.rand(2, 3, 4)
print(x)
print(x[0])
print(x[0, :, :])
print(x[0, :, :].shape)

# 矩阵乘法
# 第一个矩阵的列数必须等于第二个矩阵的行数
# 矩阵乘法计算是通过 行与列的点积
# 结果矩阵的行数 = a1.shape，列数 = a2.shape
a1 = torch.randn(2, 3)
a2 = torch.randn(3, 5)
print(a1)
print(a2)

print(torch.matmul(a1, a2))

# 向量乘法运算
# pytorch中的dot是向量的乘法，必须是一维向量
x1 = torch.rand(5)
x2 = torch.rand(5)
print(x1)
print(x2)
print(x1.dot(x2))
# 计算过程：对应位置相乘后求和，数学表示为
print((x1 * x2).sum())
