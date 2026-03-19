"""
v2.0 Multi-Agent 查询工作流

架构：Router Agent → 并行搜索 Agent → Synthesizer Agent
- Router Agent：意图分析，决定走哪些搜索策略
- Knowledge Agent：复用 v1 知识库检索（向量+HyDE+RRF）
- WebSearch Agent：复用 v1 联网搜索（MCP）
- Synthesizer Agent：合并多源结果，重排，生成答案
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph

from config.lm_config import lm_config
from processor.query_processor.base import NodeBase
from processor.query_processor.nodes.node_answer_output import NodeAnswerOutput
from processor.query_processor.nodes.node_item_name_confirm import NodeItemNameConfirm
from processor.query_processor.nodes.node_rerank import NodeRerank
from processor.query_processor.nodes.node_rrf import NodeRrf
from processor.query_processor.nodes.node_search_embedding import NodeSearchEmbedding
from processor.query_processor.nodes.node_search_embedding_hyde import NodeSearchEmbeddingHyde
from processor.query_processor.nodes.node_web_search_mcp import NodeWebSearchMcp
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.json_format_utils import serialize_json
from utils.llm_utils import get_llm_client
from utils.task_utils import add_done_task

# ==================== Router Agent ====================

ROUTER_SYSTEM_PROMPT = """你是一个智能查询路由专家。
分析用户的查询意图，决定使用哪些搜索策略来获取最准确的答案。

路由策略：
- "knowledge_only"：用户明确询问某个具体产品/设备/操作，答案在知识库中
- "web_search_only"：用户询问通用知识、新闻、时事等，需要联网搜索
- "knowledge_and_web"：用户询问产品但可能需要补充信息，同时查知识库和网络

对于 knowledge_only 或 knowledge_and_web，如果有明确的实体名称，提取并放入 entities 字段。
对于 web_search_only，返回子查询列表用于网络搜索。

请严格按照以下 JSON 格式返回，不要有多余内容：
{
    "strategy": "knowledge_only|web_search_only|knowledge_and_web",
    "entities": ["实体名1", "实体名2"],
    "sub_queries": ["子查询1", "子查询2"]
}"""

ROUTER_USER_PROMPT = """用户查询：{query}
历史对话：{history_text}

请分析查询意图并返回路由决策。"""

ROUTER_PROMPT_TEMPLATE = """你是一个智能查询路由专家。
分析用户的查询意图，决定使用哪些搜索策略来获取最准确的答案。

路由策略：
- "knowledge_only"：用户明确询问某个具体产品/设备/操作，答案在知识库中
- "web_search_only"：用户询问通用知识、新闻、时事等，需要联网搜索
- "knowledge_and_web"：用户询问产品但可能需要补充信息，同时查知识库和网络

对于 knowledge_only 或 knowledge_and_web，如果有明确的实体名称，提取并放入 entities 字段。
对于 web_search_only，返回子查询列表用于网络搜索。

请严格按照以下 JSON 格式返回，不要有多余内容：
{
    "strategy": "knowledge_only|web_search_only|knowledge_and_web",
    "entities": ["实体名1", "实体名2"],
    "sub_queries": ["子查询1", "子查询2"]
}

用户查询：{query}
历史对话：{history_text}

请分析查询意图并返回路由决策。"""


class AgentRouter(NodeBase):
    """
    Router Agent：意图分析，决定搜索策略
    """
    name: str = "agent_router"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        try:
            query = state.get("original_query")
            history = state.get("history") or []

            # 构建历史记录文本
            history_text = ""
            for msg in history:
                role = msg.get("role", "user")
                content = msg.get("text", "")
                history_text += f"{role}: {content}\n"

            # 调用 LLM 进行意图分析
            llm = get_llm_client(json_mode=True)
            prompt = ROUTER_PROMPT_TEMPLATE.format(
                query=query,
                history_text=history_text[:2000]  # 限制历史长度
            )
            messages = [
                SystemMessage(content=ROUTER_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]

            response = llm.invoke(messages)
            content = response.content

            # 解析 JSON 结果
            result = self._parse_router_response(content)
            logger.info(f"Router Agent 决策: strategy={result.get('strategy')}, entities={result.get('entities')}")

            # 更新状态
            state["router_strategy"] = result.get("strategy", "knowledge_only")
            state["router_entities"] = result.get("entities", [])
            state["router_sub_queries"] = result.get("sub_queries", [])
            state["router_raw"] = content

            return state

        except Exception as e:
            logger.error(f"Router Agent 执行异常: {e}", exc_info=True)
            # 降级：默认走知识库策略
            state["router_strategy"] = "knowledge_only"
            state["router_entities"] = state.get("item_names", [])
            state["router_sub_queries"] = []
            return state

    def _parse_router_response(self, content: str) -> dict:
        """解析 Router 的 JSON 响应"""
        import json
        try:
            # 清理可能的代码围栏
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"Router JSON 解析失败: {e}, content={content}")
            return {"strategy": "knowledge_only", "entities": [], "sub_queries": []}


# ==================== Knowledge Agent ====================

class AgentKnowledge(NodeBase):
    """
    Knowledge Agent：知识库检索（并行执行）
    整合向量搜索、HyDE、RRF 重排
    """
    name: str = "agent_knowledge"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        try:
            session_id = state.get("session_id")
            is_stream = state.get("is_stream")

            # 1. 并行执行向量搜索和 HyDE 搜索
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_search = executor.submit(self._search_embedding, state)
                future_hyde = executor.submit(self._search_hyde, state)

                embedding_chunks = future_search.result()
                hyde_chunks = future_hyde.result()

            # 2. RRF 融合
            state["embedding_chunks"] = embedding_chunks
            state["hyde_embedding_chunks"] = hyde_chunks
            rrf_result = self._rrf_merge(state)

            # 3. 标记完成
            add_done_task(session_id, self.name, is_stream)
            logger.info(f"Knowledge Agent 完成: 向量搜索={len(embedding_chunks)}, HyDE={len(hyde_chunks)}, RRF={len(rrf_result)}")

            return {"rrf_chunks": rrf_result}

        except Exception as e:
            logger.error(f"Knowledge Agent 执行异常: {e}", exc_info=True)
            session_id = state.get("session_id")
            add_done_task(session_id, self.name, state.get("is_stream"))
            return {"rrf_chunks": []}

    def _search_embedding(self, state: QueryGraphState) -> list:
        node = NodeSearchEmbedding()
        return node.process(state).get("embedding_chunks", [])

    def _search_hyde(self, state: QueryGraphState) -> list:
        node = NodeSearchEmbeddingHyde()
        return node.process(state).get("hyde_embedding_chunks", [])

    def _rrf_merge(self, state: QueryGraphState) -> list:
        node = NodeRrf()
        result = node.process(state)
        return result.get("rrf_chunks", [])


# ==================== WebSearch Agent ====================

class AgentWebSearch(NodeBase):
    """
    WebSearch Agent：联网搜索（并行执行）
    """
    name: str = "agent_web_search"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        try:
            session_id = state.get("session_id")
            is_stream = state.get("is_stream")

            # 使用 Router 的子查询或原始查询
            sub_queries = state.get("router_sub_queries", [])
            if not sub_queries:
                sub_queries = [state.get("rewritten_query", state.get("original_query", ""))]

            # 并发执行多个子查询
            all_web_docs = []
            with ThreadPoolExecutor(max_workers=len(sub_queries)) as executor:
                futures = {
                    executor.submit(self._search_one, query, session_id, is_stream): query
                    for query in sub_queries
                }
                for future in futures:
                    try:
                        docs = future.result()
                        all_web_docs.extend(docs)
                    except Exception as e:
                        logger.warning(f"WebSearch 子查询失败: {futures[future]}, error={e}")

            add_done_task(session_id, self.name, is_stream)
            logger.info(f"WebSearch Agent 完成: 返回 {len(all_web_docs)} 条结果")

            return {"web_search_docs": all_web_docs}

        except Exception as e:
            logger.error(f"WebSearch Agent 执行异常: {e}", exc_info=True)
            session_id = state.get("session_id")
            add_done_task(session_id, self.name, state.get("is_stream"))
            return {"web_search_docs": []}

    def _search_one(self, query: str, session_id: str, is_stream: bool) -> list:
        """执行单个子查询"""
        node = NodeWebSearchMcp()
        state = {
            "session_id": session_id,
            "is_stream": is_stream,
            "rewritten_query": query
        }
        result = node.process(state)
        return result.get("web_search_docs", [])


# ==================== Synthesizer Agent ====================

class AgentSynthesizer(NodeBase):
    """
    Synthesizer Agent：合并多源结果，重排，生成答案
    """
    name: str = "agent_synthesizer"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        try:
            session_id = state.get("session_id")
            is_stream = state.get("is_stream")

            # 1. 重排（如果已有 rrf_chunks 和 web_search_docs）
            if state.get("rrf_chunks") or state.get("web_search_docs"):
                rerank_node = NodeRerank()
                state = rerank_node.process(state)

            # 2. 生成答案
            answer_node = NodeAnswerOutput()
            state = answer_node.process(state)

            add_done_task(session_id, self.name, is_stream)
            logger.info(f"Synthesizer Agent 完成: answer_len={len(state.get('answer', ''))}")

            return state

        except Exception as e:
            logger.error(f"Synthesizer Agent 执行异常: {e}", exc_info=True)
            session_id = state.get("session_id")
            add_done_task(session_id, self.name, state.get("is_stream"))
            return state


# ==================== v2.0 主工作流 ====================

class KBQueryWorkflowV2:
    """
    v2.0 Multi-Agent 知识库查询工作流

    流程：
    1. Router Agent → 分析意图，决定策略
    2. Knowledge Agent / WebSearch Agent → 并行搜索（根据策略选择）
    3. Synthesizer Agent → 合并结果，生成答案
    """

    def __init__(self):
        self.workflow = StateGraph(QueryGraphState)
        self._init_nodes()
        self._register_nodes()
        self._setup_routes()
        self._compiled_app: Optional[object] = None

    def _init_nodes(self):
        """初始化所有 Agent 节点"""
        # 复用 v1 节点
        self.node_item_name_confirm = NodeItemNameConfirm()
        self.node_answer_output = NodeAnswerOutput()

        # v2 新增 Agent
        self.agent_router = AgentRouter()
        self.agent_knowledge = AgentKnowledge()
        self.agent_web_search = AgentWebSearch()
        self.agent_synthesizer = AgentSynthesizer()

    def _register_nodes(self):
        """注册节点到工作流"""
        # v1 节点
        self.workflow.add_node("node_item_name_confirm", self.node_item_name_confirm)
        self.workflow.add_node("node_answer_output", self.node_answer_output)

        # v2 Agent 节点
        self.workflow.add_node("agent_router", self.agent_router)
        self.workflow.add_node("agent_knowledge", self.agent_knowledge)
        self.workflow.add_node("agent_web_search", self.agent_web_search)
        self.workflow.add_node("agent_synthesizer", self.agent_synthesizer)

        # 虚拟节点用于路由
        self.workflow.add_node("node_multi_search", lambda x: x)
        self.workflow.add_node("node_join", lambda x: {})

    def _setup_routes(self):
        """设置工作流路由规则"""
        # 入口：实体确认
        self.workflow.set_entry_point("node_item_name_confirm")

        # 实体确认后 → Router Agent
        self.workflow.add_conditional_edges(
            "node_item_name_confirm",
            self._route_after_item_name_confirm,
            {
                "node_answer_output": "node_answer_output",
                "agent_router": "agent_router"
            }
        )

        # Router → 根据策略路由到对应 Agent
        self.workflow.add_conditional_edges(
            "agent_router",
            self._route_after_router,
            {
                "agent_knowledge": "agent_knowledge",
                "agent_web_search": "agent_web_search",
                "agent_synthesizer": "agent_synthesizer"  # 无搜索需求直接合成
            }
        )

        # 并行搜索完成后汇合到 Synthesizer
        self.workflow.add_edge("agent_knowledge", "agent_synthesizer")
        self.workflow.add_edge("agent_web_search", "agent_synthesizer")

        # Synthesizer 完成后 → 答案输出
        self.workflow.add_edge("agent_synthesizer", "node_answer_output")
        self.workflow.add_edge("node_answer_output", END)

    def _route_after_item_name_confirm(self, state: QueryGraphState) -> str:
        """实体确认后的路由"""
        if state.get("answer"):
            return "node_answer_output"
        return "agent_router"

    def _route_after_router(self, state: QueryGraphState) -> str:
        """Router 后的路由：根据策略决定搜索方式"""
        strategy = state.get("router_strategy", "knowledge_only")

        if strategy == "web_search_only":
            return "agent_web_search"
        elif strategy == "knowledge_only":
            return "agent_knowledge"
        elif strategy == "knowledge_and_web":
            # 同时走两个 Agent，使用虚拟节点分叉
            self.workflow.add_edge("agent_router", "node_multi_search")
            self.workflow.add_edge("node_multi_search", "agent_knowledge")
            self.workflow.add_edge("node_multi_search", "agent_web_search")
            return "node_multi_search"

        return "agent_knowledge"

    def compile(self):
        """编译工作流"""
        if not self._compiled_app:
            self._compiled_app = self.workflow.compile()
        return self._compiled_app

    def run(self, initial_state: QueryGraphState, stream: bool = False) -> QueryGraphState:
        """
        执行工作流
        :param initial_state: 初始状态
        :param stream: 是否流式输出
        :return: 执行完成后的状态
        """
        if not self._compiled_app:
            self.compile()

        if stream:
            return self._compiled_app.stream(initial_state)
        else:
            return self._compiled_app.invoke(initial_state)


if __name__ == "__main__":
    # 测试 v2.0 工作流
    init_state = {
        "session_id": "test_v2_session_001",
        "original_query": "HAK180烫金机如何调节温度？",
        "is_stream": False
    }

    workflow = KBQueryWorkflowV2()
    result = workflow.run(init_state, stream=False)
    logger.info(serialize_json(result, indent=4))

    # 打印图结构
    logger.info(workflow.compile().get_graph().draw_ascii())
