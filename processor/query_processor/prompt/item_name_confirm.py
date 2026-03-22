# processor/query_processor/prompt/item_name_confirm.py

ITEM_NAME_EXTRACT_SYSTEM_PROMPT = "你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"

ITEM_NAME_EXTRACT_TEMPLATE = """
历史会话：
{history_text}

当前用户问题：
{query}

请根据当前用户问题，提取用户**正在询问**的商品名称（item_names）。
1. 只提取当前问题中明确提到的商品名称，不要从历史会话中提取历史商品。
2. 如果用户使用了代词（如"这个"、"它"），请结合历史会话指代消解，确定商品名称。
3. 如果无法确定商品名称，item_names 返回空列表。
4. 请重新改写用户的问题（rewritten_query），使其成为包含商品名称的独立完整问题。

重要：只提取当前问题相关的商品，不要把历史中提到的所有商品都提取出来。

请直接返回JSON格式结果，格式如下：
{{
    "item_names": ["商品A"],
    "rewritten_query": "关于商品A，..."
}}
"""