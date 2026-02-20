import json
import logging
import re
import os

from minio import Minio

from config.minio_config import minio_config
from processor.import_processor.base import setup_logging

setup_logging()

try:
    # 1.创建客户端链接对象
    minio_client = Minio(
        endpoint=minio_config.endpoint,
        access_key=minio_config.access_key,
        secret_key=minio_config.secret_key,
        #是否强制启用HTTPS的加密链接方式，False：使用http；True：使用https
        secure=False
    )

    # 2.判断bucket是否存在，不存在则创建
    found = minio_client.bucket_exists(minio_config.bucket_name)
    if not found:
        minio_client.make_bucket(minio_config.bucket_name)

    # 3. 定义当前bucket的访问权限
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": "*"},
                "Action": "s3:GetObject",
                "Resource": f"arn:aws:s3:::{minio_config.bucket_name}/*",
            },
        ],
    }
    # 4. 设置当前bucket的访问权限
    minio_client.set_bucket_policy(minio_config.bucket_name, json.dumps(policy))

except Exception as e:
    logging.error(f"MinIO连接失败，错误原因：{e}")

def get_minio_client():
    return minio_client


def sanitize_filename(filename):
    """清理文件名中的特殊字符"""
    if not filename:
        return "unnamed_file"
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255-len(ext)] + ext
    return filename


MINIO_CHUNK_SIZE = 10 * 1024 * 1024
MINIO_MAX_RETRIES = 3

def optimized_upload(client, bucket_name, object_name, file_path):
    """优化的大文件上传，使用分片上传和重试机制"""
    import os
    import time
    for attempt in range(MINIO_MAX_RETRIES):
        try:
            file_size = os.path.getsize(file_path)
            if file_size > MINIO_CHUNK_SIZE:
                with open(file_path, 'rb') as file_data:
                    client.put_object(
                        bucket_name, object_name, file_data,
                        length=file_size,
                        content_type="application/octet-stream"
                    )
            else:
                client.fput_object(bucket_name, object_name, file_path)
            return True
        except Exception as e:
            if attempt == MINIO_MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)
    return False


if __name__ == '__main__':
    get_minio_client()