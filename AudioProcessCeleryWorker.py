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


logging.basicConfig(level=logging.INFO)

celery_app = make_celery()

# 获取全部角色信息
model_names = list_models()


class TaskResult(pydantic.BaseModel):
    code: int
    message: str


class OfficalSpeakerTTSRequest(pydantic.BaseModel):
    task_id: str
    callback_url: Optional[str] = None
    speaker: str
    language: str
    text: str


async def request_callback(method: str, url: str, **kwargs):
    async with request(method, url, **kwargs) as resp:
        return resp.ok


@celery_app.task(name="async/gpt_sovits_generate_voice", bind=True, time_limit=20)
def celery_generate_voice(task: Task, raw: dict):
    """合成语音, 成功则上传到oss

    Returns:
        _type_: Dict错误信息或者上传后的url
    """
    try:
        req = OfficalSpeakerTTSRequest.model_validate(raw)
    except pydantic.ValidationError as e:
        return TaskResult(code=400, message=e.json()).model_dump()

    if req.speaker not in model_names:
        return TaskResult(
            code=404,
            message=f"Not found {req.speaker}.pls choise one from {model_names}",
        ).model_dump()

    language = req.language
    language = language if language in ["ja", "zh", "en", "auto"] else "auto"

    # 合成音频
    try:
        audio_bytes = generate_voice(
            model=req.speaker, text=req.text, text_language=language
        )
        hash_str = hashlib.md5(audio_bytes).hexdigest()
        key = f"tts/task/{hash_str}"
        url = put_object(key, audio_bytes)
        try:
            if req.callback_url is not None:
                cb = request_callback(
                    "POST",
                    req.callback_url,
                    json={"task_id": req.task_id, "url": url},
                )
                asyncio.run(cb)
        except Exception as e:
            print(e)
        return url
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
