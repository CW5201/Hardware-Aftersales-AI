"""
processor/agent_processor/nodes/retrieval.py

Retrieval Node：把 Triage 的产物接给 RAG Tool，把 Evidence 写回 State。

职责（本阶段）：
  - 读 AgentState["triage"]
  - 用 build_tool_args_from_triage() 把 TriageResult 映射成 Tool 入参
    （纯映射，不再调 LLM）
  - invoke search_knowledge_base Tool
  - 把结构化 Evidence 写回 AgentState["retrieval"]

严格契约：
  - 不 import utils/milvus_utils，不重写 embedding / hybrid / HyDE / RRF /
    rerank / cliff cutoff —— 全部走 Tool → RetrievalService → v1 检索链路。
  - 区分两种"无结果"：
      A. error=None, documents=[]  → 检索正常完成，但知识库确实没找到
      B. error!=None               → 检索系统异常，不能当成"没有答案"
  - strategy 边界校验：即使 Triage 已保证合法，仍做一次 SUPPORTED_STRATEGIES
    白名单校验（纵深防御）。
"""

from tool.logger import logger

from core.retrieval.adaptive import pick_retrieval_strategy, SUPPORTED_STRATEGIES
from processor.agent_processor.state import AgentState
from processor.agent_processor.tools.search_knowledge_base import (
    build_tool_args_from_triage,
    get_search_service,
    make_search_knowledge_base_tool,
)


class RetrievalNode:
    """LangGraph 节点：triage → search_knowledge_base → evidence。"""

    def __init__(self, tool=None, service=None):
        # tool/service 可注入，便于单测 mock；默认用全局共享单例
        self._tool = tool if tool is not None else make_search_knowledge_base_tool(service=service)
        self._service = service if service is not None else get_search_service()

    def __call__(self, state: AgentState) -> AgentState:
        triage = state.get("triage")
        if triage is None:
            logger.warning("RetrievalNode: state.triage 缺失，跳过检索")
            return {
                **state,
                "retrieval": {
                    "documents": [],
                    "scores": [],
                    "strategy": None,
                    "metadata": {},
                    "latency": 0.0,
                    "error": "missing_triage",
                },
            }

        # 1) 从 Triage 映射 Tool 入参（不调 LLM）
        tool_args = build_tool_args_from_triage(
            user_query=state.get("user_query", ""),
            product_model=triage.product_model,
            retrieval_strategy=triage.retrieval_strategy,
        )

        # 1b) Adaptive Retrieval（Step 13）：根据 triage 复杂度 / 已有证据质量
        #     覆盖 triage 推荐的 strategy。首轮（retrieval 尚未跑过）时，
        #     简单问题 → hybrid（而非 triage 默认的 hybrid_rrf_rerank），
        #     复杂问题 → hybrid_rrf_rerank，知识不足/低置信 → HyDE + web 搜索建议。
        decision = pick_retrieval_strategy(triage, retrieval=None)
        if decision.strategy != tool_args.get("retrieval_strategy"):
            logger.info(
                f"RetrievalNode(Adaptive): strategy 覆盖 {tool_args.get('retrieval_strategy')} → "
                f"{decision.strategy}（{decision.reason}）"
            )
            tool_args["retrieval_strategy"] = decision.strategy

        # 2) 边界校验：strategy 必须落在 SUPPORTED_STRATEGIES
        strategy = tool_args.get("retrieval_strategy")
        if strategy not in SUPPORTED_STRATEGIES:
            logger.warning(
                f"RetrievalNode: 非法 strategy={strategy!r}，纠偏到默认 hybrid_rrf_rerank"
            )
            strategy = "hybrid_rrf_rerank"
            tool_args["retrieval_strategy"] = strategy

        # 3) 调 Tool（底层 = RetrievalService；Tool 已做 空 query / 异常 处理）
        try:
            raw = self._tool.invoke(tool_args)
        except Exception as e:
            # 兜底：理论上 Tool 已处理异常，这里再保险一层，避免 Graph 挂掉
            logger.exception(f"RetrievalNode: Tool invoke 异常: {e}")
            return {
                **state,
                "retrieval": {
                    "documents": [],
                    "scores": [],
                    "strategy": strategy,
                    "metadata": {},
                    "latency": 0.0,
                    "error": f"tool_failed: {type(e).__name__}: {e}",
                },
            }

        # 4) 归一化 Tool 的 dict 输出 → 结构化 evidence 字段
        evidence = {
            "documents": raw.get("documents") or [],
            "scores": [d.get("score") for d in (raw.get("documents") or [])],
            "strategy": raw.get("strategy", strategy),
            "sources": sorted(
                {d.get("source") for d in (raw.get("documents") or []) if d.get("source")}
            ),
            "metadata": raw.get("metadata") or {},
            "latency": raw.get("latency", 0.0),
            "error": raw.get("error"),
            # Adaptive Retrieval（Step 13）：本轮决策 + 是否建议 web 搜索兜底
            "adaptive": decision.to_dict(),
        }
        # 5) 区分"正常无结果"与"检索故障"（仅用于日志与上层判断，不静默吞掉）
        if evidence["error"]:
            logger.warning(
                f"RetrievalNode: 检索异常（不是无结果）: {evidence['error']}"
            )
        elif not evidence["documents"]:
            logger.info("RetrievalNode: 检索正常完成，但无匹配文档")

        return {**state, "retrieval": evidence}
