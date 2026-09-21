# config/bailian_mcp_config.py

# 导入核心依赖：数据类、环境变量读取、路径处理
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


# 定义mcp的服务配置
@dataclass
class McpConfig:
    mcp_base_url: str
    api_key : str

mcp_config = McpConfig(
    mcp_base_url=os.getenv("MCP_DASHSCOPE_BASE_URL"),
    # MCP WebSearch 走 DashScope，需要 DashScope 的 key。
    # .env 里 MCP_API_KEY 是专用 key（sk- 开头），优先用它；
    # 仅在未配置时才回退到 OPENAI_API_KEY（后者是智谱 key，喂给 DashScope 会 401）。
    api_key=os.getenv("MCP_API_KEY") or os.getenv("OPENAI_API_KEY")
)