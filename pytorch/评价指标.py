# 加载一个具体的评价指标
from datasets import load_metric

metric = load_metric(path='glue', config_name='mrpc')

predictions = [0, 1]
references = [0, 1]
metric.compute(predictions=predictions, references=references)

# 单纯使用准确率
metric = load_metric('accuracy')



