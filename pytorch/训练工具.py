
# # 加载tokenizer
# from transformers import AutoTokenizer
#
# tokenizer = AutoTokenizer.from_pretrained(
#     'C:/Users/YuT/.cache/huggingface/hub/models--hfl--rbt3/snapshots/0aa0527ff4170f29e1dfd3eb6ef60dc67e1bf75c')
#
# # 从磁盘加载中文数据集
# from datasets import load_from_disk
#
# dataset = load_from_disk('D:/Python_Code/book_translator/Pytorch/ChnSentiCorp/')
#
# # # 缩小数据规模, 便于测试.
# dataset['train'] = dataset['train'].shuffle().select(range(2000))
# dataset['test'] = dataset['test'].shuffle().select(range(100))
#
# if __name__ == '__main__':
#     def f(data, tokenizer):
#         return tokenizer(data['text'], truncation=True)
#
#
#     dataset = dataset.map(f, batched=True,
#                           batch_size=1000,
#                           num_proc=4,
#                           remove_columns=['text'],
#                           fn_kwargs={'tokenizer': tokenizer})
#
#     # 删掉太长的句子
#     def f(data):
#         return [len(i) <= 512 for i in data['input_ids']]
#
#
#     dataset = dataset.filter(f, batched=True, batch_size=1000, num_proc=4)


from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained('C:/Users/YuT/.cache/huggingface/hub/models--hfl--rbt3/snapshots/0aa0527ff4170f29e1dfd3eb6ef60dc67e1bf75c', num_labels=2)

print(model)
