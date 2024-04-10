import io
import json
import os
import re
from AudioProcess import list_models, generate_voice
import pydantic
from typing import Any, List, Dict, Optional
import logging
import time
from celery_utils import make_celery
import torch
import multiprocessing as mp

logging.basicConfig(level=logging.INFO)

# 创建Celery 实例
celery_app = make_celery()

# 获取全部角色信息
model_names = list_models()


class TaskResult(pydantic.BaseModel):
    code: int
    message: str


@celery_app.task(name="gpt_sovits_generate_voice", bind=True, time_limit=20)
def celery_generate_voice(self, speaker_name: str, text: str, language: str):
    """合成语音

    Args:
        speaker_name (str): 角色名称
        text (str): 文本内容
        language (str): 语言

    Returns:
        _type_: mp3二进制数据或者Dict错误信息
    """

    # 增加speaker_name的检测逻辑
    if not isinstance(speaker_name, str) or len(speaker_name) == 0:
        return TaskResult(
            code=400, message="Invalid speaker name provided"
        ).model_dump()

    if speaker_name not in model_names:
        return TaskResult(
            code=404,
            message=f"Not found {speaker_name}.pls choise one from {model_names}",
        ).model_dump()

    # 增加language检测逻辑，只允许 ja|zh|en|auto 这四种情况
    if language not in ["ja", "zh", "en", "auto"]:
        language = "auto"
    # 合成音频
    try:
        audio_bytes = generate_voice(
            model=speaker_name, text=text, text_language=language
        )
        return audio_bytes
    except Exception as e:
        return TaskResult(
            code=500, message=f"Failed to generate voice: {str(e)}"
        ).model_dump()


@celery_app.task(name="gpt_sovits_support_voice_info", bind=True, time_limit=20)
def get_support_voice_info(self):
    """
    Return: ["角色名称1","角色名称2"]
    """
    try:
        return model_names
    except Exception as e:
        raise self.retry(exc=e, countdown=3, max_retries=3)
