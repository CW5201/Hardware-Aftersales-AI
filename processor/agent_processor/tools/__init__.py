"""
processor/agent_processor/tools

v2 Agent 可调用的业务工具（Agent-facing Adapter）。

当前只有 search_knowledge_base —— 把 Step 1 的 RetrievalService
包装成 LangChain/LangGraph Tool，供未来的 Diagnosis/Retrieval Agent 调用。
"""
