FROM registry.aliyuncs.com/library/python:3.11
WORKDIR /app

# Debian系统切换阿里源加速
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# 清华源安装uv
RUN pip install uv -i https://pypi.tuna.tsinghua.edu.cn/simple

# 先复制依赖文件，实现构建缓存（重点）
COPY pyproject.toml uv.lock ./

# uv拉取依赖使用清华源
RUN uv sync --no-dev --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 最后复制业务代码
COPY . .

RUN mkdir -p /app/temp-files

EXPOSE 8000 8001

# 同时启动导入服务和查询服务
CMD bash -c "python -m web.api.import_service & python -m web.api.query_service & wait"