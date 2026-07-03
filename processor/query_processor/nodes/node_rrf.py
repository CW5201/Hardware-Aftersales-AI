# processor/query_processor/nodes/node_rrf.py
from typing import List, Tuple, Dict

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.json_format_utils import serialize_json
from utils.task_utils import add_done_task


class NodeRrf(NodeBase):
    """
    节点功能：Reciprocal Rank Fusion
    将多路召回的结果（向量、HyDE）进行加权融合排序。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_rrf"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 1. 获取各搜索路的搜索结果
        embedding_chunks_list = state.get("embedding_chunks") or []
        hyde_embedding_chunks_list = state.get("hyde_embedding_chunks") or []

        # 2. 清洗数据
        # embedding_search_list = [
        #     {"chunk_id": "chunk_1", "content": "向量搜索结果#1"},
        #     {"chunk_id": "chunk_2", "content": "向量搜索结果#2"},
        #     {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}
        # ]
        # embedding_search_list = []
        # for doc in embedding_chunks_list:
        #     if isinstance(doc, dict):
        #         embedding_search_list.append(doc.get("entity"))

        embedding_search_list = [doc.get("entity") for doc in embedding_chunks_list if isinstance(doc, dict)]
        hyde_embedding_search_list = [doc.get("entity") for doc in hyde_embedding_chunks_list if isinstance(doc, dict)]

        # rrf_inputs = [
        #     {"list":embedding_search_list, "weight":1.0},
        #     {"list":hyde_embedding_search_list, "weight":1.0}
        # ]

        # 3. 构建融合前的数据结构
        rrf_inputs = [
            (embedding_search_list, 1.0),
            (hyde_embedding_search_list, 1.0)
        ]

        # 4. 根据RRF公式进行数据的融合
        rrf_merge_results = self._rrf_merge(rrf_inputs, max_results=5)

        # 5. 将分数清洗出去（下一个步骤重新排序，不参考这个步骤的分数）
        rrf_chunks = [doc for doc, _ in rrf_merge_results]

        state["rrf_chunks"] = rrf_chunks

        add_done_task(state.get("session_id"), self.name, state.get("is_stream"))
        return state

    def _rrf_merge(self, rrf_inputs, k: int = 60, max_results: int = None) -> List[Tuple[Dict, float]]:
        """
        RRF (Reciprocal Rank Fusion) 融合算法
        :param rrf_inputs: 输入数据列表，每个元素是一个元组，包含一个列表和一个权重
        :param k: RRF中的平滑参数
        :param max_results: 最大结果数量限制，如果是None，则获取所有结果
        :return: 融合后的结果列表 [({},得分),({},得分),...]
        """

        # 1. 遍历需要融合的数据，并组装数据
        chunk_data = {}
        chunk_scores = {}
        for rrf_input, weight in rrf_inputs:
            # print(rrf_input)
            # print(weight)
            # rank：文档排名
            # docker：文档
            for rank, doc in enumerate(rrf_input, start=1):
                chunk_id = doc.get("chunk_id")
                # 计算每个文档的得分，并将得分组织在list中
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + weight / (k + rank)
                # 将所有文档组织在list中
                chunk_data.setdefault(
                    chunk_id,
                    doc
                )

        # 2. 将得分和文档组装在一起
        unsorted_results = [(chunk_data[chunk_id], score) for chunk_id, score in chunk_scores.items()]

        # 3. 按照得分降序排列
        sorted_results = sorted(unsorted_results, key=lambda x:x[1], reverse=True)

        return sorted_results[:max_results] if max_results else sorted_results

if __name__ == '__main__':

    # 模拟两路检索结果
    mock_state = {
        "embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#3"}},
        ]
    }

    node_rrf = NodeRrf()
    result = node_rrf(mock_state)
    logger.info(serialize_json(result, indent=4))


def normalize_rrf_scores(scores, k=60):
    """修复RRF分数归一化"""
    if not scores:
        return []
    if k <= 0:
        k = 60
    return [1.0 / (k + rank) for rank, _ in enumerate(sorted(scores, reverse=True), 1)]


def optimized_rrf_fusion(embedding_chunks, hyde_chunks, k=60):
    """优化的RRF融合算法"""
    all_chunks = embedding_chunks + hyde_chunks
    scores = normalize_rrf_scores(range(len(all_chunks)), k)
    return all_chunks