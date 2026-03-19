from langgraph.constants import END
from langgraph.graph import StateGraph

from processor.query_processor.nodes.node_answer_output import NodeAnswerOutput
from processor.query_processor.nodes.node_item_name_confirm import NodeItemNameConfirm
from processor.query_processor.nodes.node_rerank import NodeRerank
from processor.query_processor.nodes.node_rrf import NodeRrf
from processor.query_processor.nodes.node_search_embedding import NodeSearchEmbedding
from processor.query_processor.nodes.node_search_embedding_hyde import NodeSearchEmbeddingHyde
from processor.query_processor.nodes.node_web_search_mcp import NodeWebSearchMcp
from processor.query_processor.state import QueryGraphState
from tool.logger import logger

class KBQueryWorkflow:

    def __init__(self):
        """初始化工作流：创建状态图、注册节点、定义路由规则"""
        # 1. 初始化LangGraph状态图
        self.workflow = StateGraph(QueryGraphState)
        # 2. 初始化所有业务节点（实例属性，支持多实例隔离）
        self._init_nodes()
        # 3. 注册节点到工作流
        self._register_nodes()
        # 4. 设置入口和路由规则
        self._setup_routes()
        # 5. 编译工作流（懒加载，首次执行时编译）
        self._compiled_app = None

    def _init_nodes(self):
        """初始化所有业务节点（私有方法，封装节点创建逻辑）"""
        self.node_item_name_confirm = NodeItemNameConfirm()
        self.node_search_embedding = NodeSearchEmbedding()
        self.node_search_embedding_hyde = NodeSearchEmbeddingHyde()
        self.node_web_search_mcp = NodeWebSearchMcp()
        self.node_rrf = NodeRrf()
        self.node_rerank = NodeRerank()
        self.node_answer_output = NodeAnswerOutput()

    def _register_nodes(self):
        """注册所有节点到工作流（私有方法，统一管理节点注册）"""
        # 节点标识与实例属性名保持一致，便于维护
        self.workflow.add_node("node_item_name_confirm", self.node_item_name_confirm)  # 确认主体
        self.workflow.add_node("node_multi_search", lambda x: x)  # 虚拟节点：多路搜索分叉点（状态不变）
        self.workflow.add_node("node_search_embedding", self.node_search_embedding)  # 向量搜索
        self.workflow.add_node("node_search_embedding_hyde", self.node_search_embedding_hyde)  # 假设性答案向量搜索
        self.workflow.add_node("node_web_search_mcp", self.node_web_search_mcp)  # 联网搜索
        self.workflow.add_node("node_join", lambda x: {})  # 虚拟节点：多路搜索合并点
        self.workflow.add_node("node_rrf", self.node_rrf)  # 排序
        self.workflow.add_node("node_rerank", self.node_rerank)  # 重排
        self.workflow.add_node("node_answer_output", self.node_answer_output)  # 生成

    def _setup_routes(self):
        """设置工作流路由规则"""
        # 1、设置入口节点
        self.workflow.set_entry_point("node_item_name_confirm")

        # 2、注册条件路由边
        self.workflow.add_conditional_edges(
            "node_item_name_confirm",
            self._route_after_item_name_confirm,
            {
                "node_answer_output": "node_answer_output",
                "node_multi_search": "node_multi_search"
            }
        )

        # 3. 并发执行搜索
        self.workflow.add_edge("node_multi_search", "node_search_embedding")
        self.workflow.add_edge("node_multi_search", "node_search_embedding_hyde")
        self.workflow.add_edge("node_multi_search", "node_web_search_mcp")

        # 4. 多路搜索结果合并
        self.workflow.add_edge("node_search_embedding", "node_join")
        self.workflow.add_edge("node_search_embedding_hyde", "node_join")
        self.workflow.add_edge("node_web_search_mcp", "node_join")

        # 5. 合并 -> 排序 -> 重排 -> 生成 -> 结束
        self.workflow.add_edge("node_join", "node_rrf")
        self.workflow.add_edge("node_rrf", "node_rerank")
        self.workflow.add_edge("node_rerank", "node_answer_output")
        self.workflow.add_edge("node_answer_output", END)

    def _route_after_item_name_confirm(self, state: QueryGraphState) -> str:
        """主体名称确认后的条件路由函数"""
        if state.get("answer"):
            return "node_answer_output"

        # 否则继续搜索流程
        return "node_multi_search"

    def compile(self):
        """编译工作流（公开方法，支持手动触发编译）"""
        if not self._compiled_app:
            self._compiled_app = self.workflow.compile()
        return self._compiled_app

    def run(self, initial_state: QueryGraphState, stream: bool = False) -> QueryGraphState:
        """
        统一执行入口，支持切换invoke/stream
        :param initial_state:  初始状态对象
        :param stream: 是否是流式输出
        :return: 执行完成后的状态对象
        """
        """"""
        if not self._compiled_app:
            self.compile()
        if stream:
            return self._compiled_app.stream(initial_state)
        else:
            return self._compiled_app.invoke(initial_state)


if __name__ == "__main__":

    # 初始化图状态
    init_state = {
        "session_id": "session_002",
        "original_query": "怎么调节180烫金机的温度"
    }


    # 创建流程对象
    workflow = KBQueryWorkflow()
    # for chunk in workflow.run(init_state, stream=True):
    #     logger.debug(chunk)

    for chunk in workflow.run(init_state, stream=False):
        logger.debug(chunk)

    logger.info(workflow.compile().get_graph().draw_ascii())


def optimized_query_workflow(session_id, query):
    """优化的查询工作流"""
    if not session_id or not query:
        return None
    return {"session_id": session_id, "query": query}