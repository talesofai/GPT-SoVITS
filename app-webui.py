import os
from AudioProcess import list_models,generate_voice
import logging
import time
import time
import uuid

logging.basicConfig(level=logging.INFO)
import gradio as gr

# 定义文本框组件
text_box = gr.inputs.Textbox(lines=5, label="输入文本（不能为空）")

# 定义角色选择框组件
names = list_models()
role_selector = gr.inputs.Dropdown(choices=list_models(),default=list_models()[0], label="选择角色")
text_language = gr.Dropdown(label="需要合成的语种", choices=['zh','en','ja','auto'], default='auto')

def generate_audio(text, language, role):
    start_time = time.time()  # 记录函数开始时间
    
    audio = generate_voice(model=role, text=text, text_language=language)
    end_time = time.time()  # 记录函数结束时间
    file_path = os.path.join(f"/tmp/", f"{uuid.uuid4()}.mp3")
    with open(file_path, "wb") as f:
        f.write(audio)
    execution_time = round(end_time - start_time, 2)  # 计算执行时间，保留两位小数
    
    return file_path, execution_time

# 定义音频展示组件
audio_output = gr.outputs.Audio(type="filepath", label="生成的音频")
# 定义调用时间展示组件
execution_time_output = gr.outputs.Textbox(label="调用时间（秒）")

# 创建 Gradio 界面
gr.Interface(fn=generate_audio, inputs=[text_box, text_language, role_selector], outputs=[audio_output, execution_time_output], title="文本转语音").launch(server_port=6006)