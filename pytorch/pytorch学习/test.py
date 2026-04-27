# import torch
# print('CUDA available:', torch.cuda.is_available())
# print('CUDA device count:', torch.cuda.device_count())
# if torch.cuda.is_available():
#     print('GPU:', torch.cuda.get_device_name(0))
#
#
# 最小测试脚本
import os

print("开始测试...")
print("当前目录:", os.getcwd())

dataset_path = 'D:/Python_Code/book_translator/Pytorch'
print(f"路径存在: {os.path.exists(dataset_path)}")

print("测试完成！")