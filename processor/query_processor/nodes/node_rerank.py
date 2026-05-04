# processor/query_processor/nodes/node_rerank.py
from typing import List, Dict, Any

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.json_format_utils import serialize_json
from utils.reranker_http_utils import rerank_documents
from utils.task_utils import add_done_task

# -----------------------------
# Rerank / TopK 全局常量
# -----------------------------
# 动态 TopK 硬上限：最多取前 N 条（<=10）
RERANK_MAX_TOPK: int = 5
# 最小 TopK：至少保留前 N 条（>=1，且 <= RERANK_MAX_TOPK）
RERANK_MIN_TOPK: int = 2 #总数最少条数

# 断崖阈值（绝对，判断高分文档）
RERANK_GAP_ABS: float = 0.3
# 断崖阈值（相对，判断低分文档）
RERANK_GAP_RATIO: float = 0.25

class NodeRerank(NodeBase):
    """
    节点功能：使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_rerank"

    def normalize_scores(self, scores):
        """修复重排分数归一化错误"""
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [0.5] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        执行重排序
        流程: 合并多源文档 → Reranker 计算相关性 → 断崖检测动态截断
        :param state: 需包含 rrf_chunks、web_search_docs、rewritten_query
        :return: 更新后的 state，包含 reranked_docs
        """


        # 1. 合并多源文档
        merged_multi_docs: List[Dict[str, Any]] = self._step_1_merge_multi_source_docs(state)

        # 2. Rerank 精排(精排打分)
        reranked_docs: List[Dict[str, Any]] = self._step_2_rerank_merged_docs(state, merged_multi_docs)

        # 3. 动态 Top_K 截取(断崖检测)
        cutoff_docs = self._step_3_cliff_cutoff(reranked_docs)

        # 4. 更新state
        state['reranked_docs'] = cutoff_docs

        # 5. 返回state
        add_done_task(state.get("session_id"), self.name, state.get("is_stream"))
        return state

    def _step_1_merge_multi_source_docs(self, state: QueryGraphState) -> List[Dict[str, Any]]:
        """合并本地 RRF 结果和网络搜索结果为统一格式"""

        rrf_chunks = state.get("rrf_chunks")
        web_search_docs = state.get("web_search_docs")
        final_docs = []

        for rrf_doc in rrf_chunks:
            final_rrf_doc = {
                "chunk_id": rrf_doc.get("chunk_id"),
                "title": rrf_doc.get("item_name"),
                "content": rrf_doc.get("content"),
                "url": None,
                "source": "local"
            }
            final_docs.append(final_rrf_doc)

        for web_doc in web_search_docs:
            final_web_doc = {
                "chunk_id": None,
                "title": web_doc.get("title"),
                "content": web_doc.get("snippet"),
                "url": web_doc.get("url"),
                "source": "web"
            }
            final_docs.append(final_web_doc)

        return final_docs

    def _step_2_rerank_merged_docs(self, state: QueryGraphState, merged_multi_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """使用 Reranker 模型对文档进行精排"""

        # 1. 获取改写后的问题
        user_query = state.get("rewritten_query")

        # 2. 清洗数据
        contents = [doc.get("content") for doc in merged_multi_docs]

        # 3. 使用 Reranker 模型对文档进行相关性得分的计算
        rerank_scores = rerank_documents(user_query, contents)

        # 4. 将文档和对应的分数组装到一个字典列表中
        # reranked_docs = [
        #     {
        #         "chunk_id": doc.get("chunk_id"),
        #         "title": doc.get("title"),
        #         "content": doc.get("content"),
        #         "url": doc.get("url"),
        #         "source": doc.get("source"),
        #         "score": score
        #     }
        #     for doc, score in zip(merged_multi_docs, rerank_scores)
        # ]
        reranked_docs = [{**doc, "score": score} for doc, score in zip(merged_multi_docs, rerank_scores)]

        # 5. 按分数由高到低排序
        reranked_docs = sorted(reranked_docs, key=lambda x: x["score"], reverse=True)

        # 6. 返回结果
        return reranked_docs

    def _step_3_cliff_cutoff(self, ranked_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """断崖检测截断：相邻得分差距超过阈值时截断。"""

        if not ranked_docs:
            return []

        # 计算的动态topK的动态上限
        upper_bound = min(RERANK_MAX_TOPK, len(ranked_docs))
        # 计算的动态topK的动态下限
        lower_bound = min(RERANK_MIN_TOPK, upper_bound)

        # 初步定义截断的位置
        cutoff_pos = upper_bound

        for idx in range(lower_bound - 1, upper_bound - 1):
            current_score = ranked_docs[idx].get("score")
            next_score = ranked_docs[idx + 1].get("score")

            if current_score is None or next_score is None:
                continue

            # 计算相邻文档的分数差
            abs_gap = current_score - next_score
            rel_gap = abs_gap / (abs(current_score) + 1e-6)

            if abs_gap >= RERANK_GAP_ABS or rel_gap >= RERANK_GAP_RATIO:
                cutoff_pos = idx + 1
                break

        return ranked_docs[:cutoff_pos]

if __name__ == "__main__":

    mock_state = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        "rrf_chunks": [
            {
                "chunk_id": "local_1",
                "item_name": "主板维修手册",
                "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"
            },
            {
                "chunk_id": "local_2",
                "item_name": "闲聊",
                "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"
            },
        ],
        "web_search_docs": [
            {
                "url": "https://example.com/repair",
                "title": "短路查修指南",
                "snippet": "主板通电前先打各主供电电感对地阻值，阻值偏低就是短路。"
            },
            {
                "url": "https://example.com/news",
                "title": "科技新闻",
                "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"
            },
        ],
    }

    node_rerank = NodeRerank()
    result = node_rerank(mock_state)
    logger.info(serialize_json(result, indent=4))