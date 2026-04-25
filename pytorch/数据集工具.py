from datasets import load_dataset
# dataset = load_dataset('seamew/ChnSentiCorp')


# ChnSentiCorp暂时在huggingface上下架了, 从本地加载
from datasets import load_from_disk

dataset = load_from_disk('ChnSentiCorp/')
# print(dataset)

# 把数据保存到本地
# dataset.save_to_disk('./data/ChnSentiCorp')

# 数据集基本操作

dataset = dataset['train']


# print(dataset[1])

# for i in range(10):
#     print(dataset[i])

# print(dataset['label'][:10])

# 排序
# sorted_dataset = dataset.sort('label')
# print(sorted_dataset['label'][:10])
#
# print(sorted_dataset['label'][-10:])

# 打乱数据
# shuffled_dataset = dataset.shuffle(seed=10)
#
# print(shuffled_dataset['label'][:10])

# 数据抽样, 可以实现数据抽样
# dataset.select([0, 10, 20, 30, 40, 50])
# print(dataset.select)

# 数据过滤
# def f(data):
#     return data['text'].startswith('非常不错')

# print(dataset.filter(f)) # 对数据集中的每个样本调用函数 f

# 训练测试集划分
# dataset.train_test_split(test_size=0.1)


# 数据分桶: 把数据均匀的分成N份
# 分成4份, 取出索引为0 的那一份
# dataset.shard(num_shards=4, index=0)


# 重命名字段
# dataset.rename_column('text', 'text_rename')


# 删除字段
# dataset.remove_columns(['text'])


# 映射函数
# def f(data):
#     data['text'] = 'My sentence: ' + data['text']
#     return data
#
#
# maped_dataset = dataset.map(f)
#
# print(dataset['text'][:5])
#
# print(maped_dataset['text'][:5])


# 批处理加速
# def f(data):
#     text = data['text']
#     text = ['My sentence: ' + i for i in text]
#     data['text'] = text
#     return data
#
#
# maped_dataset = dataset.map(function=f,
#                             batched=True,
#                             batch_size=1000,
#                             num_proc=4)
#
# print(dataset['text'][0])
# print(maped_dataset['text'][0])

# 设置数据格式, 会直接修改原始数据
# dataset.set_format(type='torch', columns=['label'], output_all_columns=True)

# 保存为其他格式
# 保存成csv文件
# dataset.to_csv(path_or_buf='ChnSentiCorp.csv')
# # 从csv文件加载数据
# csv_dataset = load_dataset(path='csv', data_files='ChnSentiCorp.csv', split='train')
# print(csv_dataset[20])

# 保存成json文件
dataset.to_json(path_or_buf='ChnSentiCorp.json')

# 读取json数据
json_dataset = load_dataset(path='json', data_files='ChnSentiCorp.json', split='train')
print(json_dataset[10])
