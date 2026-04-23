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

# 基本的编码函数
out = tokenizer.encode(
    text=sents[0],
    text_pair=sents[1],
    # 句子太长就截断到max_length
    truncation=True,
    # 句子不够长就padding到max_length的长度
    padding='max_length',
    add_special_tokens=True,
    max_length=25,
    return_tensors=None
)
print(out)

# 把数字还原成字符串
tokenizer.decode(out)
print(tokenizer.decode(out))
