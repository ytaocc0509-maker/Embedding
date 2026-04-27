# sentence = 'hello everyone , today is a good day .'
#
# # 字典， 也叫做词表 vocabulary
# vocab = {
#     '<SOS>': 0,
#     '<EOS>': 1,
#     'hello': 2,
#     'everyone': 3,
#     'today': 4,
#     'is': 5,
#     'a': 6,
#     'good': 7,
#     'day': 8,
#     ',': 9,
#     '.': 10
# }
#
# sent = '<SOS> ' + sentence + ' <EOS>'
# print(sent)
#
# # 英文分词， 比较简单， 直接按照空格区分就可以。
# # 中文可以使用分词工具，比如jieba分词。
# words = sent.split()
# print(words)
#
# print([vocab[i] for i in words])

# 1.使用编码器
# 模型和它的编码器是成对使用的， 你使用什么模型， 就它提供的编码器。
# 编码器的名字一般和模型的名字是一样的。 
# bert-base-chinese
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path='bert-base-chinese',
    # cache_dir=None,
    # force_download=False
)

sents = [
    '你站在桥上看风景',
    '看风景的人在楼上看你',
    '明月装饰了你的窗子',
    '你装饰了别人的梦'
]

# # 基本的编码函数
# out = tokenizer.encode(
#     text=sents[0],
#     text_pair=sents[1],
#     # 句子太长就截断到max_length
#     truncation=True,
#     # 句子不够长就padding到max_length的长度
#     padding='max_length',
#     add_special_tokens=True,
#     max_length=25,
#     return_tensors=None
# )
# print(out)

# # 把数字还原成字符串
# tokenizer.decode(out)
# print(tokenizer.decode(out))

# 进阶版编码函数
# out = tokenizer.encode_plus(
#     text=sents[0],
#     text_pair=sents[1],
#     truncation=True,
#     padding='max_length',
#     # 句子最大长度
#     max_length=25,
#     # 是否添加特殊字符, [cls]
#     add_special_tokens=True,
#     # 返回的数据类型, 默认返回列表, 可以返回Tensorflow, pytorch的tensor
#     return_tensors=None,
#     # 0, 1 表示是哪个句子的数据
#     return_token_type_ids=True,
#     # 有用部分标1, pad部分标0
#     return_attention_mask=True,
#     # 特殊字符标1, 其他位置标0
#     return_special_tokens_mask=True,
#     # 返回句子长度.
#     return_length=True
# )
#

# 批量编码函数
# out = tokenizer.batch_encode_plus(
#     # 句子对
#     # 如果要对单句子编码, batch_text_or_text_pairs=[sents[0], sents[1], sents[2], ...]
#     batch_text_or_text_pairs=[(sents[0], sents[1]), (sents[2], sents[3])],
#     truncation=True,
#     padding='max_length',
#     max_length=25,
#     add_special_tokens=True,
#     return_tensors=None,
#     return_token_type_ids=True,
#     return_attention_mask=True,
#     return_special_tokens_mask=True,
#     return_length=True
# )
#
# for k, v in out.items():
#     print(k, ': ', v)
#
#
#     # 字典的操作
# # 获取字典
# vocab = tokenizer.get_vocab()
# print(vocab)
#
# print(len(vocab))

# 添加新词
tokenizer.add_tokens(new_tokens=['明月', '装饰', '窗子'])

# 添加特殊字符
tokenizer.add_special_tokens({'eos_token': '[EOS]'})

for word in ['明月', '装饰', '窗子', '[EOS]']:
    print(tokenizer.get_vocab()[word])

# 用新词表去编码
out = tokenizer.encode(text='明月装饰了你的窗子[EOS]',
                       text_pair=None,
                       truncation=True,
                       padding='max_length',
                       add_special_tokens=True,
                       max_length=10,
                       return_tensors=None)
print(out)

tokenizer.decode(out)


# 总结: 编码器工作流程: 定义字典, 句子预处理, 分词, 编码

