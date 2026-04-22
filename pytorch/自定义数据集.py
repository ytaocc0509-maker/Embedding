import torch
import numpy as np
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

import glob
from PIL import Image

all_img_path = glob.glob(r'./dataset/*.jpg')

if not all_img_path:
    print("Error: No images found in ./dataset directory")
    exit()

# 建立类别和索引之间的映射关系
species = ['cloudy', 'rain', 'shine', 'sunrise']

species_to_idx = dict((c, i) for i, c in enumerate(species))

# 调换一下key和value的顺序
idx_to_species = dict((v, k) for k, v in species_to_idx.items())

# 生成所有图片的label
all_labels = []

for img in all_img_path:
    label_found = False
    for i, c in enumerate(species):
        if c in img:
            all_labels.append(i)
            label_found = True
            break
    if not label_found:
        print(f"Warning: No label found for {img}")
        all_labels.append(-1)

# 借助ndarray的索引取值的方法, 打乱数据
index = np.random.permutation(len(all_img_path))

all_img_path = np.array(all_img_path)[index]

all_labels = np.array(all_labels)[index]

# 手动的划分一下训练数据和测试数据
split = int(len(all_img_path) * 0.8)

train_imgs = all_img_path[:split]
train_labels = all_labels[:split]

test_imgs = all_img_path[split:]
test_labels = all_labels[split:]

transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor()
])


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, labels, transform):
        # 调用父类的init这里可以不写.
        #         super().__init__()
        self.imgs = img_paths # 图片路径
        self.labels = labels # 标签
        self.transforms = transform # 图像预处理变换
 
    def __getitem__(self, index):
        # 根据index获取item
        img_path = self.imgs[index]
        label = self.labels[index]

        # 通过PIL的Image读取图片，并确保转换为RGB格式
        img = Image.open(img_path).convert('RGB')
        data = self.transforms(img)
        return data, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.imgs)


train_ds = MyDataset(train_imgs, train_labels, transform)
test_ds = MyDataset(test_imgs, test_labels, transform)

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=16 * 2)


# 添加BN层.
# 定义模型
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)  # 16 * 94 * 94
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)  # 16 * 47 * 47

        self.conv2 = nn.Conv2d(16, 32, 3)  # 32 * 45 * 45  -> pooling -> 32 * 22 * 22
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3)  # 64 * 20 * 20  -> pooling -> 64 * 10 * 10
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout()

        # batch , channel, height, width, 64,
        self.fc1 = nn.Linear(64 * 10 * 10, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 4)

    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.conv1(x)
        if batch_size > 1:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        if batch_size > 1:
            x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = nn.Flatten()(x)
        x = self.fc1(x)
        if batch_size > 1:
            x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        if batch_size > 1:
            x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = Net()
# 把model拷到gpu上
if torch.cuda.is_available():
    model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()


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
