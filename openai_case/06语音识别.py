import base64
from openai import OpenAI
import os
import pathlib
from dotenv import load_dotenv
import dashscope

# 加载环境变量
load_dotenv()
try:
    # 请替换为实际的音频文件路径
    file_path = "推销.wav"
    # 请替换为实际的音频文件MIME类型
    audio_mime_type = "audio/mpeg"

    file_path_obj = pathlib.Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"音频文件不存在: {file_path}")

    base64_str = base64.b64encode(file_path_obj.read_bytes()).decode()
    data_uri = f"data:{audio_mime_type};base64,{base64_str}"

    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        # 以下为北京地域url
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

#     stream_enabled = False  # 是否开启流式输出
#     completion = client.chat.completions.create(
#         model="qwen3-asr-flash",
#         messages=[
#             {
#                 "content": [
#                     {
#                         "type": "input_audio",
#                         "input_audio": {
#                             "data": data_uri
#                         }
#                     }
#                 ],
#                 "role": "user"
#             }
#         ],
#         stream=stream_enabled,
#         # stream设为False时，不能设置stream_options参数
#         # stream_options={"include_usage": True},
#         extra_body={
#             "asr_options": {
#                 # "language": "zh",
#                 "enable_itn": False
#             }
#         }
#     )
#     if stream_enabled:
#         full_content = ""
#         print("流式输出内容为：")
#         for chunk in completion:
#             # 如果stream_options.include_usage为True，则最后一个chunk的choices字段为空列表，需要跳过（可以通过chunk.usage获取 Token 使用量）
#             print(chunk)
#             if chunk.choices and chunk.choices[0].delta.content:
#                 full_content += chunk.choices[0].delta.content
#         print(f"完整内容为：{full_content}")
#     else:
#         print(f"非流式输出内容为：{completion.choices[0].message.content}")


# ----------------音频输入 ----------------
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": data_uri,
                        "format": "wav",
                    },
                }
            ],
        }
    ]

    completion = client.chat.completions.create(
        model="qwen3-livetranslate-flash",
        messages=messages,
       # modalities=["text", "audio"],
       # audio={"voice": "Cherry", "format": "wav"},
        stream=False,
       # stream_options={"include_usage": True},
        extra_body={"translation_options": {"source_lang": "zh", "target_lang": "en"}},
    )

    # for chunk in completion:
    #     print(chunk)
    print(f"非流式输出内容为：{completion.choices[0].message.content}")

    en_text = completion.choices[0].message.content

# 文字转成配音
    dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

    #text = "那我来给大家推荐一款T恤，这款呢真的是超级好看，这个颜色呢很显气质，而且呢也是搭配的绝佳单品，大家可以闭眼入，真的是非常好看，对身材的包容性也很好，不管啥身材的宝宝呢，穿上去都是很好看的。推荐宝宝们下单哦。"
    # SpeechSynthesizer接口使用方法：dashscope.audio.qwen_tts.SpeechSynthesizer.call(...)
    response = dashscope.MultiModalConversation.call(
        # 如需使用指令控制功能，请将model替换为qwen3-tts-instruct-flash
        model="qwen3-tts-flash",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        text=en_text,
        voice="Cherry",
        language_type="English",  # 建议与文本语种一致，以获得正确的发音和自然的语调。
        # 如需使用指令控制功能，请取消下方注释，并将model替换为qwen3-tts-instruct-flash
        # instructions='语速较快，带有明显的上扬语调，适合介绍时尚产品。',
        # optimize_instructions=True,
        stream=False
    )
    print(response)


except Exception as e:
    print(f"错误信息：{e}")


