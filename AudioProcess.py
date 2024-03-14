import concurrent.futures
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
# 只在cuda12.1版本及以上启用flash_attn
cuda_version = torch.version.cuda
major_version, minor_version = map(int, cuda_version.split('.')[:2])

flash_attn_enabled = False
if major_version > 12 or (major_version == 12 and minor_version >= 1):
    print("CUDA版本号大于等于12.1,启动flash_attn")
    flash_attn_enabled = True


# 配置上启用cuda和flash_atten以及半精度
default_config = {
    "device": "cuda",
    "is_half": True,
    "t2s_weights_path": "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
    "vits_weights_path": "GPT_SoVITS/pretrained_models/s2G488k.pth",
    "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
    "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    "flash_attn_enabled": flash_attn_enabled
}

tts_config = TTS_Config({"default": default_config, "custom": default_config})
tts_config.device = device
tts_config.is_half = True
tts_pipline = TTS(tts_config)       # 初始化批处理
gpt_path = tts_config.t2s_weights_path
sovits_path = tts_config.vits_weights_path

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

#-------------------------------
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

def normalize_string(input_string:str) -> str:
    # 删除特殊标点符号
    input_string = input_string.replace('、','，')
    special_characters = r'[@#¥%&*（）——+=-]'
    normalized_string = re.sub(special_characters, '', input_string)
    
    special_characters = r'[【】「」\[\]\\\|｜；;‘“\'"<>/《》；‘]'
    normalized_string = re.sub(special_characters, '', normalized_string)
    
    # 替换 !! 和 ... 为 ,
    normalized_string = re.sub(r'!', '.', normalized_string)
    normalized_string = re.sub(r'~', '.', normalized_string)
    normalized_string = re.sub(r'～', '.', normalized_string)
    normalized_string = re.sub(r'\.\.+', ',', normalized_string)
    normalized_string = normalized_string.replace("…",',')
    normalized_string = re.sub(r',+','.',normalized_string)

    # 删除空格和换行符和制表符
    normalized_string = normalized_string.replace('\n', '.').replace('\r', '').replace('\t', '')

    return normalized_string

def generate_voice(model, text, text_language) -> bytes:
    """
    合成文本为音频
    """
    if model not in model_infos:
        raise ValueError(f"Invalid model name {model}.")
    model_info = model_infos[model]

    # 将文本规范化
    text = normalize_string(text)

    # 切换/加载模型
    tts_pipline.init_t2s_weights(model_info.gpt_path)
    tts_pipline.init_vits_weights(model_info.sovits_path)
    
    # 正式推理之前预处理参考文本
    tts_pipline.prompt_cache["ref_audio_path"] = model_info.refer_wav_path
    tts_pipline.set_ref_audio(model_info.refer_wav_path)

    gen = inference(text=text,
                    text_lang=text_language,
                    ref_audio_path=model_info.refer_wav_path,
                    prompt_text=model_info.refer_text,
                    prompt_lang=model_info.refer_text_language,
                    batch_size=32)  # 经过测试batch_szie = [16,32] 最好
    sampling_rate, audio_data = next(gen)
    audio_io = BytesIO()
    sf.write(audio_io, audio_data, sampling_rate, format="mp3")
    audio_io.seek(0)
    # 获取mp3二进制数据
    binary_data = audio_io.getvalue()
    return binary_data

if __name__ == "__main__":
    model_names = list_models()
    print(model_names)
    
    output_folder = "./audio_output"
    os.makedirs(output_folder,exist_ok=True)
    for model_name in model_names:
        # 合成音频
        text = "Hello。我｜‘；们要、吃「」蛋【】糕了～不过I think得@#¥%&*（）————+...还是由你来切吧~～"
        text_language = "auto"
        audio_bytes = generate_voice(
            model=model_name, text=text, text_language=text_language)
        # 保存在本地
        output_path = os.path.join(output_folder,f"{model_name}.mp3")
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        print(f"Audio synthesized and saved to {output_path}")
