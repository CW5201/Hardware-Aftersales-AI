# processor/query_processor/nodes/node_search_embedding_hyde.py
from config.milvus_config import milvus_config
from processor.query_processor.base import NodeBase
from processor.query_processor.prompt.search_embedding_hyde import HYDE_PROMPT
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.json_format_utils import serialize_json
from utils.llm_utils import get_llm_client
from utils.milvus_utils import create_hybrid_search_requests, get_milvus_client, hybrid_search
from utils.task_utils import add_done_task


class NodeSearchEmbeddingHyde(NodeBase):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_search_embedding_hyde"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        HyDE (Hypothetical Document Embedding) 检索节点
        核心思想：通过LLM生成假设性答案（HyDE文档），将其向量化后用于检索，以解决短查询语义稀疏问题。

        执行步骤：
        1. 参数提取：从会话状态中获取改写后的查询（rewritten_query）和已确认的商品名（item_names）。
        2. 生成假设文档 (Step 1)：调用LLM，基于用户问题生成一段假设性的理想回答（即HyDE文档）。
        3. 混合检索 (Step 2)：
           - 将“用户问题 + 假设文档”合并，生成BGE-M3稠密+稀疏向量。
           - 在Milvus中执行混合检索（带商品名过滤），召回最相似的知识切片。
        4. 结果封装：返回检索到的切片列表和生成的假设文档，更新会话状态。

        :param state: 会话状态字典，包含 rewritten_query, item_names 等
        :return: 包含 hyde_embedding_chunks (检索结果) 和 hyde_doc (假设文档) 的字典
        """


        try:
            # 1、用户问题和已确认商品名
            rewritten_query = state.get("rewritten_query")
            item_names = state.get("item_names")

            # 2. 生成假设性文档
            hyde_doc = self._step_1_create_hyde_doc(rewritten_query)

            # 3、用“重写问题 + 假设文档”检索切片
            res = self._step_2_search_embedding_hyde(
                rewritten_query=rewritten_query,
                hyde_doc=hyde_doc,
                item_names=item_names
            )

            # 4. 封装返回结果
            add_done_task(state.get("session_id"), self.name, state.get("is_stream"))
            return {
                "hyde_embedding_chunks": res,
                "hyde_doc": hyde_doc
            }

        except Exception as e:
            logger.exception(f"向量搜索失败: {e}")
            return {}


    def _step_1_create_hyde_doc(self, rewritten_query: str) -> str:
        """
        阶段1：利用大模型根据用户查询生成假设性文档（Hypothetical Document）。
        HyDE的核心在于：利用LLM生成一个“虚构但相关”的文档，用该文档的向量去检索真实的文档，
        从而缓解短查询（Query）与长文档（Document）在语义空间不匹配的问题。

        :param rewritten_query: 用户改写后的查询语句
        :return: LLM生成的假设性文档内容
        """

        try:
            llm = get_llm_client()
            gyde_prompt = HYDE_PROMPT.format(rewritten_query = rewritten_query)
            hyde_doc = llm.invoke(gyde_prompt).content

            return hyde_doc
        except Exception as e:
            logger.exception(f"LLM调用失败: {e}")
            raise e


    def _step_2_search_embedding_hyde(
            self,
            rewritten_query: str,
            hyde_doc: str,
            item_names=None
    ):
        """
        阶段2：利用“重写问题 + 假设性文档”生成 embedding，并到向量库检索切片。

        :param rewritten_query: 改写后的查询
        :param hyde_doc: Step 1 生成的假设性文档
        :param item_names: 商品名称列表，用于元数据过滤 (item_name in [...])
        :return: 检索结果列表
        """

        try:

            # 1. 拼接字符串
            combined_text = rewritten_query + " " + hyde_doc

            # 2. 生成 embedding
            embeddings = generate_embeddings([combined_text])
            dense_vector = embeddings.get("dense")[0]
            sparse_vector = embeddings.get("sparse")[0]

            # 3. 组织查询表达式：匹配已确认商品名
            # 'item_name in ["BrotherHAK-180烫金机","BrotherHAK180烫金机"]'
            expr = f'item_name in {item_names}'

            # 4. 组织混合向量检索的参数
            reqs = create_hybrid_search_requests(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                expr=expr,
                limit=10
            )

            # 5. 执行混合向量检索
            client = get_milvus_client()
            collection_name = milvus_config.chunks_collection
            search_res = hybrid_search(
                client=client,
                collection_name=collection_name,
                reqs=reqs,
                ranker_weights=(0.8, 0.2),
                output_fields=["chunk_id", "content", "item_name"]
            )

            return search_res[0] if search_res else []

        except Exception as e:
            logger.exception(f"向量搜索失败: {e}")
            raise e


def fix_hyde_prompt_format(query):
    """优化HyDE生成质量"""
    if not query:
        return ""
    query = query.strip()
    if len(query) > 500:
        query = query[:500] + "..."
    query = query.replace("\n\n", "\n")
    return query


def optimized_hyde_search(query, item_names):
    """优化的HyDE搜索"""
    fixed_query = fix_hyde_prompt_format(query)
    return {"query": fixed_query, "item_names": item_names}


if __name__ == "__main__":

    init_state = {
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？",
        "item_names": ["BrotherHAK180烫金机", "BrotherHAK-180烫金机"]
    }
    node_search_embedding_hyde = NodeSearchEmbeddingHyde()
    result = node_search_embedding_hyde(init_state)
    logger.info(serialize_json(result, indent=4))
