import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import numpy as np
import matplotlib.pyplot as plt
import torchvision

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
for train_or_test in ['train', 'test']:
    for spec in species:
        path = os.path.join(base_dir, train_or_test, spec)
        os.mkdir(path)

# 要判断一个图片属于哪个类别.
# 'cloudy' in img

# 要把dataset中的图片全部拷贝到train, test目录下的4个子目录中.
for i, img in enumerate(filenames):
    for spec in species:
        if spec in img:
            img_path = os.path.join(base_dir, img)
            if i % 5 == 0:
                path = os.path.join(base_dir, 'test', spec, img)
            else:
                path = os.path.join(base_dir, 'train', spec, img)
            shutil.copy(img_path, path)

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

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)

imgs, labels = next(iter(train_dl))

# 显示一张图片
img = imgs[0]

img = img + 1
img = img / 2

# reshape不可以, 会打乱数据. 
plt.imshow(img.permute(1, 2, 0))
