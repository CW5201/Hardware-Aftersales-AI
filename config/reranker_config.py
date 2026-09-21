# config/reranker_config.py

from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class RerankerConfig:
    text_rerank_api_key: str # DashScope API Key
    text_rerank_model: str # 模型名称
    text_rerank_instruct: str # 是否使用指令

reranker_config = RerankerConfig(
    # Rerank（qwen3-rerank）走 DashScope，需要 DashScope 的 key。
    # .env 里 RERANK_API_KEY 是专用 key（sk- 开头），优先用它；
    # 仅在未配置时才回退到 OPENAI_API_KEY（后者是智谱 key，喂给 DashScope 会 401）。
    text_rerank_api_key=os.getenv("RERANK_API_KEY") or os.getenv("OPENAI_API_KEY"),
    text_rerank_model=os.getenv("TEXT_RERANK_MODEL"),
    text_rerank_instruct=os.getenv("TEXT_RERANK_INSTRUCT")
)