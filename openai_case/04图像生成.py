import json
import os
import dashscope
from dashscope import MultiModalConversation
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 以下为北京地域url，若使用新加坡地域的模型，需将url替换为：https://dashscope-intl.aliyuncs.com/api/v1
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

messages = [
    {
        "role": "user",
        "content": [
            {"text": "一个美女在海边散步"}
        ]
    }
]

api_key = os.getenv("DASHSCOPE_API_KEY")

response = MultiModalConversation.call(
    api_key=api_key,
    model="qwen-image-2.0-pro",
    messages=messages,
    result_format='message',
    stream=False,
    watermark=False,
    prompt_extend=True,
    negative_prompt="低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。",
    size='2048*2048'
)

if response.status_code == 200:
    #print(json.dumps(response, ensure_ascii=False))
    print(response['output']['choices'][0]['message']['content'][0]['image'])
else:
    print(f"HTTP返回码：{response.status_code}")
    print(f"错误码：{response.code}")
    print(f"错误信息：{response.message}")
