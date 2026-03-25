# processor/query_processor/nodes/node_web_search_mcp.py
import asyncio
import json

from agents.mcp import MCPServerStreamableHttp

from config.bailian_mcp_config import mcp_config
from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.json_format_utils import serialize_json
from utils.task_utils import add_done_task


class NodeWebSearchMcp(NodeBase):
    """
    节点功能，调用外部搜索引擎补充信息
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_web_search_mcp"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        query = state.get("rewritten_query")
        result = asyncio.run(self._mcp_call(query))
        web_docs = []
        if result:
            json_str = result.content[0].text
            pages = json.loads(json_str).get("pages")
            for item in pages:
                snippet = item.get("snippet")
                url = item.get("url")
                title = item.get("title")

                web_docs.append({"title": title, "url": url, "snippet": snippet})

        add_done_task(state.get("session_id"), self.name, state.get("is_stream"))
        if web_docs:
            return {"web_search_docs": web_docs}
        return {}


    async def _mcp_call(self, query: str):

        # 1.创建mcp客户端对象
        mcp_client = MCPServerStreamableHttp(
            name="Streamable HTTP Python Server",
            params={
                "url": mcp_config.mcp_base_url,
                "headers": {"Authorization": f"Bearer {mcp_config.api_key}"},
                "timeout": 10,
            },
            cache_tools_list=True,
            max_retry_attempts=3,
        )

        try:
            # 2. 建立连接
            await mcp_client.connect()

            # 3. 执行查询
            result = await mcp_client.call_tool(
                tool_name="bailian_web_search",
                arguments={
                    "query": query,
                    "count": 5
                }
            )

            return result

        finally:
            # 4. 清理资源
            await mcp_client.cleanup()




if __name__ == "__main__":

    init_state = {
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？"
    }

    node_web_search_mcp = NodeWebSearchMcp()
    result = node_web_search_mcp(init_state)
    logger.info(serialize_json(result, indent=4))


def optimized_web_search(query, max_results=5):
    """优化的网络搜索"""
    if not query:
        return []
    query = query.strip()
    if len(query) > 500:
        query = query[:500]
    return {"query": query, "max_results": max_results}

