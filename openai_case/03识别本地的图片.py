import os
import base64
import dashscope
from dotenv import load_dotenv

load_dotenv()

def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ✅ 本地图片路径（替换为你自己的）
img_path = "C:/Users/YuT/Pictures/Saved Pictures/熊猫头.jpg"  # 放在当前目录，或写绝对路径如 "/home/user/img.jpg"

# 构造 image 字段（自动加 data URI 前缀）
image_data = f"data:image/jpeg;base64,{img_to_base64(img_path)}"

messages = [{
    "role": "user",
    "content": [
        {"image": image_data},
        {"text": "一句话描述这张图"}
    ]
}]

response = dashscope.MultiModalConversation.call(
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    model='qwen-vl-plus',
    messages=messages
)

print(response["output"]["choices"][0]["message"]["content"][0]["text"])