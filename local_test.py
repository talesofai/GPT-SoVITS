from io import BytesIO
import json
import logging
import re
import torch
import soundfile as sf
import os
import time
import sys
import random
from typing import Dict, List
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append(os.path.join(now_dir, "GPT_SoVITS"))
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)

device = "cuda"

if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]

is_half = eval(os.environ.get("is_half", "True"))

print(f"device: {device}, is_half: {is_half}")

# 配置上启用cuda和flash_atten以及半精度
default_config = {
    "device": "cuda",
    "is_half": True,
    "t2s_weights_path": "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
    "vits_weights_path": "GPT_SoVITS/pretrained_models/s2G488k.pth",
    "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
    "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    "flash_attn_enabled": True
}

tts_config = TTS_Config({"default":default_config,"custom":default_config})
tts_config.device = device
tts_config.is_half = is_half
tts_pipline = TTS(tts_config)       # 先一步初始化模型
gpt_path = tts_config.t2s_weights_path
sovits_path = tts_config.vits_weights_path

# 推理


def inference(text,
              text_lang,
              ref_audio_path,
              prompt_text,
              prompt_lang,
              batch_size,
              ):

    inputs = {
        "text": text,
        "text_lang": text_lang,
        "ref_audio_path": ref_audio_path,
        "prompt_text": prompt_text,     # 必定有参考音频的文本
        "prompt_lang": prompt_lang,
        "top_k": 5,  # GPT配置部分硬编码
        "top_p": 1,
        "temperature": 1,
        "text_split_method": 'cut0',    # 不切直接放进显存里,长句可能会爆炸
        "batch_size": batch_size,        # 需要详细测试下不同的batch_size对于性能的影响
        "speed_factor": 1.0,
        "split_bucket": False,            # 数据分桶会降低计算量,太短的句子提升不明显
        "return_fragment": False,        # 不启用流式返回
    }
    return tts_pipline.run(inputs)

# ------------------------------------
# 管理模型


class ModelBasicInfo:
    # 运行模型必要信息
    gpt_path: str
    sovits_path: str
    refer_wav_path: str
    refer_text: str
    refer_text_language: str

    def __init__(self, gpt_path: str, sovits_path: str, refer_wav_path: str, refer_text: str):
        self.gpt_path = gpt_path
        self.sovits_path = sovits_path
        self.refer_wav_path = refer_wav_path

        # 解析 refer_text，去除换行符和制表符
        refer_text = refer_text.replace("\n", "").replace("\t", "")
        # 使用 "|" 切分 text 和 language
        parts = refer_text.split("|")
        if len(parts) == 2:
            self.refer_text_language, self.refer_text = parts
        else:
            raise ValueError(
                f"Invalid format in {refer_text}. Expected format: 'language|text'.")


def search_models(root_dir: str) -> Dict[str, ModelBasicInfo]:
    try:
        model_info = {}
        for root, dirs, files in os.walk(root_dir):
            if "gpt.ckpt" in files and "sovits.ckpt" in files:
                model_folder = os.path.basename(root)
                gpt_path = os.path.join(root, "gpt.ckpt")
                sovits_path = os.path.join(root, "sovits.ckpt")
                refer_wav_path = os.path.join(root, "reference_audio.wav")
                refer_text_path = os.path.join(root, "reference_audio.lab")

                # 检查参考音频文件是否存在
                if not os.path.exists(refer_wav_path) or not os.path.exists(refer_text_path):
                    logging.warning(
                        f"Model '{model_folder}' is missing reference audio files.")
                    continue

                # 假设参考文本文件是纯文本文件，直接读取内容
                with open(refer_text_path, 'r') as f:
                    refer_text = f.read()

                model_info[model_folder] = ModelBasicInfo(
                    gpt_path, sovits_path, refer_wav_path, refer_text)
    except Exception as e:
        raise Exception(f"some thing erorr on {model_folder}:{e}")

    return model_info

jokes = [
    "为什么程序员总是冷静？因为他们有很多漏洞可以修复。",
    "有一天，我对我的电脑说：“你是我的最好朋友！”然后它蓝屏了。",
    "为什么程序员不喜欢去海滩？因为他们害怕沙盒。",
    "为什么数据库管理员是最好的心理医生？因为他们总是处理关系问题。",
    "为什么程序员喜欢喝凉水？因为他们喜欢“冷静”。",
    "为什么前端开发人员总是穿着凉鞋？因为他们喜欢在浏览器中“脱鞋”。",
    "为什么程序员总是饿着肚子？因为他们喜欢“循环”。",
    "为什么程序员喜欢猫？因为它们有九条“线程”。",
    "为什么程序员不喜欢赌博？因为他们不喜欢“风险”。",
    "为什么程序员不喜欢打电话？因为他们害怕“接口”。"
]

# 使用os.getenv获取环境变量MODEL_FOLDER的值，如果未设置，则默认为"/root/autodl-tmp/"
model_folder = os.getenv('MODEL_FOLDER', "/root/autodl-tmp/models/")
model_infos = search_models(model_folder)

# 找到本地所有模型
def list_models() -> List[str]:
    """
    列出找到的模型列表
    """
    global model_infos
    model_infos = search_models(model_folder)
    return list(model_infos.keys())

# # 初始化模型
# model_name = random.choice(list_models())
# print(f"choise character: {model_name}")
# model_info = model_infos[model_name]

# # 切换/加载模型
# tts_pipline.init_t2s_weights(model_info.gpt_path)
# tts_pipline.init_vits_weights(model_info.sovits_path)

# # text = "你说的对，但是原神是由米哈游自主研发的一款全新开放世界冒险游戏"
# text_language = "zh"

# # 正式推理之前预处理参考文本
# tts_pipline.prompt_cache["ref_audio_path"] = model_info.refer_wav_path
# tts_pipline.set_ref_audio(model_info.refer_wav_path)

# for _ in range(10):
#     gen = inference(text=generate_test_word(),
#                     text_lang=text_language,
#                     ref_audio_path=model_info.refer_wav_path,
#                     prompt_text=model_info.refer_text,
#                     prompt_lang=model_info.refer_text_language,
#                     batch_size=16)

#     # 需要将ndarray转成音频二进制
#     sampling_rate, audio_data = next(gen)
#     wav_io = BytesIO()
#     sf.write(wav_io, audio_data, sampling_rate, format="mp3")
#     wav_io.seek(0)
#     # 获取 wav 二进制数据
#     binary_data = wav_io.getvalue()
#     # 测试结束

performance_stats = {}  # 存储性能统计信息

for model_name in list_models():
    model_info = model_infos[model_name]
    total_time = 0
    min_time = float("inf")
    max_time = float("-inf")

    # 切换/加载模型
    tts_pipline.init_t2s_weights(model_info.gpt_path)
    tts_pipline.init_vits_weights(model_info.sovits_path)

    # 正式推理之前预处理参考文本
    tts_pipline.prompt_cache["ref_audio_path"] = model_info.refer_wav_path
    tts_pipline.set_ref_audio(model_info.refer_wav_path)

    for i in range(16):
        start_time = time.time()
        # 以下是原来的代码
        gen = inference(text=jokes[i%len(jokes)],
                        text_lang='zh',
                        ref_audio_path=model_info.refer_wav_path,
                        prompt_text=model_info.refer_text,
                        prompt_lang=model_info.refer_text_language,
                        batch_size=16)

        sampling_rate, audio_data = next(gen)
        wav_io = BytesIO()
        sf.write(wav_io, audio_data, sampling_rate, format="mp3")
        wav_io.seek(0)
        binary_data = wav_io.getvalue()
        # 测试结束
        end_time = time.time()

        elapsed_time = end_time - start_time
        total_time += elapsed_time
        min_time = min(min_time, elapsed_time)
        max_time = max(max_time, elapsed_time)

    average_time = total_time / 16

    performance_stats[model_name] = {
        "average_time": average_time,
        "min_time": min_time,
        "max_time": max_time
    }

# 输出 Markdown 格式的表格
print("| 角色 | 平均时间 | 最短时间 | 最长时间 |\n|---|---|---|---|")
for model_name, stats in performance_stats.items():
    print(f"| {model_name} | {stats['average_time']} | {stats['min_time']} | {stats['max_time']} |")