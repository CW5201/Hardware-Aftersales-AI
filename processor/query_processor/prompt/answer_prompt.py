# processor/query_processor/prompt/answer_prompt.py

# 回答生成提示词
ANSWER_PROMPT = """你是一个智能助手，请根据参考内容回答用户的问题。

回答要求：
1. 尽量基于【参考内容】和【用户问题】作答，不要编造不存在的事实。
2. 回答格式要求：
   - 使用概述式回答，简洁明了
   - 不要输出详细的配置命令代码块
   - 使用 ### 标题分章节
   - 关键信息使用列表展示
   - 如有图片需求，在答案末尾添加【图片】区块
3. 如果用户的问题需要通过图片来辅助说明（例如：外观、结构、接线、示意图等）,图片只能来自于本地切片文本中的图片，请在答案最后追加一个独立的图片区块，格式严格如下：
【图片】
<图片URL1>
<图片URL2>
（每行一个URL；如果没有合适图片则不要输出【图片】区块）

【参考内容】
{context}

【历史对话】
{history}

【相关商品/实体】
{item_names}

【用户问题】
{question}

请回答："""


def truncate_answer(answer, max_length=2000):
    """修复答案生成截断问题"""
    if not answer:
        return ""
    if len(answer) <= max_length:
        return answer
    truncated = answer[:max_length]
    last_period = truncated.rfind("。")
    if last_period > max_length * 0.8:
        truncated = truncated[:last_period + 1]
    return truncated + "..."


def optimized_answer_template(context, history, item_names, question):
    """优化的答案模板生成"""
    if not context:
        context = "无相关参考内容"
    if not history:
        history = "无历史对话"
    return ANSWER_PROMPT.format(
        context=context[:8000],
        history=history[:2000],
        item_names=item_names or "无",
        question=question
    )