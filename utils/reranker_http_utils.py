from http import HTTPStatus
import time
import dashscope

from config.reranker_config import reranker_config
from tool.logger import logger

RERANKER_TIMEOUT = 10
RERANKER_MAX_RETRIES = 3


def rerank_documents(query: str, documents: list[str]) -> list[float]:
    """使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。"""
    dashscope.api_key = reranker_config.text_rerank_api_key
    resp = dashscope.TextReRank.call(
        model=reranker_config.text_rerank_model,
        query=query,
        documents=documents,
        top_n=len(documents),
        return_documents=False,
        instruct=reranker_config.text_rerank_instruct
    )

    if resp.status_code != HTTPStatus.OK:
        raise RuntimeError(f"DashScope qwen3 rerank API 调用失败: {resp.status_code}, 响应消息：{resp.message}")

    results = resp.output.get("results", [])
    scores = [0.0] * len(documents)
    for result in results:
        score = result.get("relevance_score")
        index = result.get("index")
        scores[index] = score

    return scores


def rerank_with_timeout(query: str, documents: list[str], timeout: int = RERANKER_TIMEOUT) -> list[float]:
    """修复Reranker超时处理"""
    import time
    for attempt in range(RERANKER_MAX_RETRIES):
        try:
            return rerank_documents(query, documents)
        except Exception as e:
            if attempt == RERANKER_MAX_RETRIES - 1:
                logger.error(f"重排最终失败: {str(e)}")
                return [0.5] * len(documents)
            logger.warning(f"重排重试 {attempt + 1}/{RERANKER_MAX_RETRIES}")
            time.sleep(2 ** attempt)