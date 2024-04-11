import os
import oss2

# from oss2.credentials import EnvironmentVariableCredentialsProvider

import dotenv

env_file = os.path.join(os.getcwd(), ".env")
if dotenv.load_dotenv(env_file):
    print(".env loaded")
else:
    raise Exception(f"{env_file} not found")

BUCKET_URL = "https://oss.talesofai.cn/"
auth = oss2.Auth(os.getenv("OSS_ACCESS_KEY_ID"), os.getenv("OSS_ACCESS_KEY_SECRET"))


def put_object(key: str, bytes: bytes):
    endpoint, bucket_name = os.getenv("OSS_ENDPOINT"), os.getenv("OSS_BUCKET")
    # 填写Bucket名称。
    bucket = oss2.Bucket(
        auth,
        endpoint=endpoint,
        bucket_name=bucket_name,
    )
    bucket.put_object(key, bytes)
    return f"{BUCKET_URL}{key}"
