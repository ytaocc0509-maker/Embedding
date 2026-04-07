import os
import dashscope
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

messages = [
    {
        "role": "user",
        "content": [
            {"image": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"},
            {"text": "这是是什么?"}
        ]
    }
]
response = dashscope.MultiModalConversation.call(
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    model='qwen-vl-max',  # 此处以qwen-vl-max为例，可按需更换模型名称。
    messages=messages
    )
print(response.output["choices"][0]["message"]["content"][0]['text'])