from datasets import load_dataset
from transformers import BertTokenizer

# 加载编码器工具
tokenizer = BertTokenizer.from_pretrained(
    'C:/Users/YuT/.cache/huggingface/hub/models--bert-base-chinese/snapshots/8f23c25b06e129b6c986331a13d8d025a92cf0ea')

# 试编码句子, 观察输出
out = tokenizer(
    text=['从明天起，做一个幸福的人。', '喂马， 劈柴，周游世界。'],
    truncation=True,
    padding='max_length',
    max_length=17,
    return_tensors='pt',
    return_length=True
)

# for k, v in out.items():
#     print(k, v.shape)

import torch

from datasets import load_from_disk


class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.dataset = load_from_disk('ChnSentiCorp')[split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['text']
        label = self.dataset[i]['label']

        return text, label


# dataset = Dataset('train')

# len(dataset)
#
# print(dataset[20])

# 定义计算机设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 数据整理函数
def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    # 编码
    data = tokenizer(text=sents,
                     truncation=True,
                     padding='max_length',
                     max_length=500,
                     return_tensors='pt',
                     return_length=True)

    # input_ids: 编码之后的数字
    # attention_mask: 0的位置是不需要计算attention的， 1的位置表示要计算attention
    # token_type_ids: token的类型， 0表示第一个句子， 1表示第二个句子。 
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    labels = torch.LongTensor(labels)

    # 把数据拷贝到计算设备上
    # 这个操作也可以在训练的时候做。 
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    token_type_ids = token_type_ids.to(device)
    labels = labels.to(device)

    return input_ids, attention_mask, token_type_ids, labels


# 测试一下整理函数
# 先模拟一批数据

data = [
    ('你站在桥上看风景', 1),
    ('看风景的人在楼上看你', 0),
    ('明月装饰了你的窗', 1),
    ('你装饰了别人的梦', 0)
]

input_ids, attention_mask, token_type_ids, labels = collate_fn(data)
input_ids.shape, attention_mask.shape, token_type_ids.shape

print(input_ids)
print(attention_mask)
print(token_type_ids)
print(labels)
