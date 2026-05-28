# utils/llm_utils.py

import time
from langchain_openai import ChatOpenAI

from config.lm_config import lm_config
from tool.logger import logger

LLM_MAX_RETRIES = 3
LLM_TIMEOUT = 30

_llm_client_cache = {}


def get_llm_client(model: str | None = None, json_mode: bool = False) -> ChatOpenAI:
    """
    获取 LangChain ChatOpenAI 客户端实例
    - model: 允许不同节点使用不同模型
    - json_mode: True 时要求输出 JSON
    """
    m = model or lm_config.llm_model
    key = (m, json_mode)
    if key in _llm_client_cache:
        return _llm_client_cache[key]

    extra_body = {"enable_thinking": False}

    model_kwargs: dict = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}

    client = ChatOpenAI(
        model=m,
        temperature=lm_config.llm_temperature,
        api_key=lm_config.api_key,
        base_url=lm_config.base_url,
        extra_body=extra_body,
        model_kwargs=model_kwargs,
    )
    _llm_client_cache[key] = client
    return client


def call_llm_with_retry(prompt, max_retries=LLM_MAX_RETRIES):
    """修复LLM超时未重试"""
    for attempt in range(max_retries):
        try:
            client = get_llm_client()
            return client.invoke(prompt, config={"timeout": LLM_TIMEOUT})
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"LLM调用失败: {str(e)}")
                raise
            wait_time = 2 ** attempt
            logger.warning(f"LLM调用重试 {attempt + 1}/{max_retries}, 等待{wait_time}秒")
            time.sleep(wait_time)


if __name__ == "__main__":
    client = get_llm_client()
    print(client)

    client = get_llm_client()
    print(client)

    client = get_llm_client("qwen-max", True)
    print(client)