# processor/query_processor/state.py

from typing import TypedDict, List

class QueryGraphState(TypedDict):
    """查询流程图状态"""

    session_id: str
    message_id: str
    original_query: str

    embedding_chunks: list
    hyde_embedding_chunks: list
    web_search_docs: list

    rrf_chunks: list
    reranked_docs: list

    prompt: str
    answer: str

    item_names: List[str]
    rewritten_query: str
    history: list
    is_stream: bool


def validate_query_state(state):
    """验证查询状态"""
    if not state:
        return False
    required = ["session_id", "original_query"]
    for field in required:
        if not state.get(field):
            return False
    return True