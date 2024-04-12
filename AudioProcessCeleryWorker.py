import asyncio
from aiohttp import request
from AudioProcess import list_models, generate_voice
import pydantic
import logging
from celery_utils import make_celery
from celery import Task
import hashlib
from aliyun_oss import put_object
from typing import Optional
from functools import partial
from base64 import b64encode

logging.basicConfig(level=logging.INFO)

celery_app = make_celery()

# 获取全部角色信息
model_names = list_models()


class TaskResult(pydantic.BaseModel):
    code: int
    message: str


class OfficalSpeakerTTSRequest(pydantic.BaseModel):
    task_uuid: str
    callback_url: str
    speaker: str
    language: str
    text: str
    format: str


async def request_callback(method: str, url: str, **kwargs):
    async with request(method, url, **kwargs) as resp:
        return resp.ok


def sync_callback(method: str, url: str, **kwargs):
    try:
        return asyncio.run(request_callback(method, url, **kwargs))
    except Exception as e:
        print(e)


@celery_app.task(name="async/gpt_sovits_generate_voice", bind=True, time_limit=20)
def celery_async_generate_voice(task: Task, raw: dict):
    """合成语音, 成功则上传到oss

    Returns:
        _type_: Dict错误信息或者上传后的url
    """
    try:
        req = OfficalSpeakerTTSRequest.model_validate(raw)
    except pydantic.ValidationError as e:
        return TaskResult(code=400, message=e.json()).model_dump()

    respond = partial(sync_callback, method="POST", url=req.callback_url)

    if req.format not in ["oss_url", "base64"]:
        respond(
            json={
                "success": False,
                "task_uuid": req.task_uuid,
                "message": "format is not supported",
            },
        )
        return False

    if req.speaker not in model_names:
        respond(
            json={
                "success": False,
                "task_uuid": req.task_uuid,
                "message": f"Not found {req.speaker}.pls choise one from {model_names}",
            },
        )
        return False

    language = req.language
    language = language if language in ["ja", "zh", "en", "auto"] else "auto"

    # 合成音频
    try:
        audio_bytes = generate_voice(
            model=req.speaker, text=req.text, text_language=language
        )
        if req.format == "base64":
            encoded = b64encode(audio_bytes).decode("utf-8")
            respond(
                json={"success": True, "task_uuid": req.task_uuid, "base64": encoded},
            )
        elif req.format == "oss_url":
            hash_str = hashlib.md5(audio_bytes).hexdigest()
            key = f"tts/task/{hash_str}"
            oss_url = put_object(key, audio_bytes)
            respond(
                json={"success": True, "task_uuid": req.task_uuid, "oss_url": oss_url},
            )
        return True
    except Exception as e:
        e = str(e)
        respond(
            json={"success": False, "task_uuid": req.task_uuid, "message": e},
        )
        return False


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
def get_support_voice_info(task: Task):
    """
    Return: ["角色名称1","角色名称2"]
    """
    try:
        return model_names
    except Exception as e:
        raise task.retry(exc=e, countdown=3, max_retries=3)
