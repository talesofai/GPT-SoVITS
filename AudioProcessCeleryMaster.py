import logging
from celery import Celery
from celery_utils import make_celery
import random
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

class TTSCallbackDto(BaseModel):
    success: bool
    task_uuid: str
    base64: Optional[str] = None
    oss_url: Optional[str] = None
    message: Optional[str] = None

test_app = FastAPI()

@test_app.post("/callback")
async def test_tts_callback(res: TTSCallbackDto):
    print(res)
    return

def start_callback_server():
    import uvicorn
    uvicorn.run(test_app, host="127.0.0.1", port=8123)


def run_celery_task(celery_app, task_name, timeout=10, queue_name="default", **kwargs):
    if kwargs == {}:
        kwargs = None
    result = celery_app.send_task(
        name=task_name, kwargs=kwargs, queue=queue_name, priority=1, time_limit=timeout
    )
    try:
        # 尝试获取任务结果，并设置超时时间10秒
        task_result = result.get()
        if isinstance(task_result, dict) and "code" in task_result:
            raise Exception(
                f"code: {task_result['code']}\terror message: {task_result['message']}"
            )
        else:
            print("The task was successful.")
    except TimeoutError:
        # 如果在指定的超时时间内任务未完成，则处理超时异常
        print(f"The task '{task_name}' timed out after {timeout} seconds.")
        return None  # 或者根据需要返回一个合适的值或抛出异常
    return task_result


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    
    from threading import Thread
    
    svr = Thread(target=start_callback_server)
    svr.start()
    
    # 创建Celery 实例
    celery_app: Celery = make_celery()

    # 调用获取支持角色信息的任务
    result_support_voice_info = run_celery_task(
        celery_app=celery_app,
        task_name="gpt_sovits_support_voice_info",
        queue_name="async/GPTSoVits",
    )

    # 获取任务结果
    support_voice_info = result_support_voice_info

    # 打印支持的角色信息
    speaker_name = random.choice(support_voice_info)
    print(f"Supported voice models: {support_voice_info}\n choise: {speaker_name}")

    # 调用生成音频的任务(async)
    audio_result = run_celery_task(
        celery_app=celery_app,
        task_name="async/gpt_sovits_generate_voice",
        queue_name="async/GPTSoVits",
        raw={
            "task_uuid": "123",
            "callback_url": "http://localhost:8123/callback",
            "speaker": speaker_name,
            "language": "zh",
            "text": "你说的对，但是《原神》是由米哈游自主研发的一款全新开放世界冒险游戏",
            "format": "base64"
        },
    )

    print(audio_result)

    # 调用生成音频的任务(sync)
    audio_result = run_celery_task(
        celery_app=celery_app,
        task_name="gpt_sovits_generate_voice",
        queue_name="async/GPTSoVits",
        speaker_name=speaker_name,
        text="你说的对，但是《原神》是由米哈游自主研发的一款全新开放世界冒险游戏",
        language="zh",
    )

    # 检测返回的结果是否为二进制文件
    if isinstance(audio_result, bytes):
        print("Voice generated successfully!")
        # 处理二进制音频文件，例如保存到文件或进行其他处理
        with open("output.wav", "wb") as audio_file:
            audio_file.write(audio_result)
    else:
        print(f"Failed to generate voice. Error message: {audio_result}")

    # 检测返回的结果是否为二进制文件
    # if isinstance(audio_result, bytes):
    #     print("Voice generated successfully!")
    #     # 处理二进制音频文件，例如保存到文件或进行其他处理
    #     with open("output.wav", "wb") as audio_file:
    #         audio_file.write(audio_result)
    # else:
    #     print(f"Failed to generate voice. Error message: {audio_result}")