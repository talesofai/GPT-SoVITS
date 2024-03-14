from celery import Celery
import os

def make_celery() -> Celery:
    dev = os.environ.get("DEV",None)
    if dev:
        broker = "redis://:Talesofai123!@127.0.0.1:6379/0"
        backend = "redis://:Talesofai123!@127.0.0.1:6379/1"
    else:
        broker = "redis://:Talesofai123!@r-uf66qbjvvp58njt3nxpd.redis.rds.aliyuncs.com:6379/10"
        backend = "redis://:Talesofai123!@r-uf66qbjvvp58njt3nxpd.redis.rds.aliyuncs.com:6379/11"
    app = Celery("gpt_vist_celery", broker=broker, backend=backend)
    return app
