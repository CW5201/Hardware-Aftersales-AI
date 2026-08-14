# processor/query_processor/prompt/search_embedding_hyde.py


HYDE_PROMPT = """
请基于以下用户查询生成一个简洁的回答范文。
用户查询: {rewritten_query}
要求：
1. 回答要简洁明了，包含核心信息即可
2. 假设你是该领域的专家，提供专业的解释
3. 不要使用"假设"、"可能"等不确定的词汇
4. 保持回答与查询主题高度相关
5. 使用中文回答且不超过300字
"""

MAX_PROMPT_TOKENS = 2000

def truncate_prompt(prompt, max_tokens=MAX_PROMPT_TOKENS):
    """优化HyDE提示词token超限处理"""
    if not prompt:
        return ""
    estimated_tokens = len(prompt) * 1.2
    if estimated_tokens <= max_tokens:
        return prompt
    truncate_ratio = max_tokens / estimated_tokens
    truncate_length = int(len(prompt) * truncate_ratio)
    return prompt[:truncate_length] + "..."