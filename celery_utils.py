from celery import Celery
import os


def make_celery() -> Celery:
    prod = os.environ.get("PROD", None)
    if prod:
        broker = "redis://:Talesofai123!@r-uf60zdlv3xuzwsec0ppd.redis.rds.aliyuncs.com:6379/10"
        backend = "redis://:Talesofai123!@r-uf60zdlv3xuzwsec0ppd.redis.rds.aliyuncs.com:6379/11"
        print("Use prod env!!!")
    else:
        broker = "redis://:Talesofai123!@r-uf66qbjvvp58njt3nxpd.redis.rds.aliyuncs.com:6379/10"
        backend = "redis://:Talesofai123!@r-uf66qbjvvp58njt3nxpd.redis.rds.aliyuncs.com:6379/11"
    app = Celery("gpt_vist_celery", broker=broker, backend=backend)
    return app
