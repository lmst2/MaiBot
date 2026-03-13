"""
MaiSaka - 了解模块
负责从对话中提取和存储用户个人特征信息。
"""

from typing import List
from knowledge_store import get_knowledge_store, KNOWLEDGE_CATEGORIES


def build_knowledge_summary() -> str:
    """
    构建了解分类摘要，用于 LLM 请求。

    Returns:
        格式化的分类列表文本
    """
    store = get_knowledge_store()
    return store.get_categories_summary()


def extract_category_ids_from_result(result: str) -> List[str]:
    """
    从 LLM 返回结果中提取分类编号。

    Args:
        result: LLM 返回的结果文本

    Returns:
        分类编号列表
    """
    if not result:
        return []

    # 检查是否表示"无相关内容"
    if any(keyword in result for keyword in ["无", "没有", "不适用", "无需", "无相关"]):
        return []

    # 解析编号（支持逗号分隔、空格分隔、换行分隔）
    category_ids = []
    for part in result.replace(",", " ").replace("，", " ").replace("\n", " ").split():
        part = part.strip()
        if part in KNOWLEDGE_CATEGORIES:
            category_ids.append(part)

    return category_ids


def format_context_for_memory(context_messages: List[dict]) -> str:
    """
    格式化上下文消息为文本，用于记忆分析。

    Args:
        context_messages: 上下文消息列表

    Returns:
        格式化后的文本
    """
    parts = []
    for msg in context_messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            parts.append(f"用户: {content}")
        elif role == "assistant":
            # 跳过感知消息
            if "【AI 感知】" not in content:
                parts.append(f"助手: {content}")

    return "\n".join(parts)


async def store_knowledge_from_context(
    llm_service,
    context_messages: List[dict],
    store_result_callback=None,
) -> int:
    """
    记忆部分：从上下文中提取并存储了解信息。

    在上下文裁切时触发：
    1. 请求 LLM 分析聊天内容涉及哪些分类
    2. 为每个分类创建 subAgent 提取相关内容
    3. 存入了解列表

    Args:
        llm_service: LLM 服务实例
        context_messages: 需要分析的上下文消息
        store_result_callback: 存储结果回调函数

    Returns:
        成功存储的了解信息数量
    """
    store = get_knowledge_store()
    context_text = format_context_for_memory(context_messages)
    categories_summary = build_knowledge_summary()

    if not context_text:
        return 0

    try:
        # 第一步：分析涉及哪些分类
        category_ids = await llm_service.analyze_knowledge_categories(context_messages, categories_summary)

        if not category_ids:
            return 0

        # 第二步：为每个分类提取内容并存储
        stored_count = 0
        for category_id in category_ids:
            try:
                # 提取该分类的相关内容
                extracted_content = await llm_service.extract_knowledge_for_category(
                    context_messages, category_id, store.get_category_name(category_id)
                )

                if extracted_content:
                    # 存储到了解列表
                    success = store.add_knowledge(
                        category_id=category_id, content=extracted_content, metadata={"source": "context_compression"}
                    )
                    if success:
                        stored_count += 1
                        if store_result_callback:
                            store_result_callback(category_id, store.get_category_name(category_id), extracted_content)
            except Exception:
                # 单个分类失败不影响其他分类
                continue

        return stored_count

    except Exception:
        return 0


async def retrieve_relevant_knowledge(
    llm_service,
    chat_history: List[dict],
) -> str:
    """
    提取部分：根据当前上下文检索相关的了解信息。

    在每次对话前触发（EQ 模块和 timing 模块位置）：
    1. 请求 LLM 分析需要哪些分类的了解内容
    2. 提取对应分类的所有内容并拼接
    3. 返回格式化后的了解内容

    Args:
        llm_service: LLM 服务实例
        chat_history: 当前对话历史

    Returns:
        格式化后的了解内容文本
    """
    store = get_knowledge_store()
    categories_summary = store.get_categories_summary()

    try:
        # 分析需要哪些分类
        category_ids = await llm_service.analyze_knowledge_need(chat_history, categories_summary)

        if not category_ids:
            return ""

        # 获取并格式化了解内容
        formatted_knowledge = store.get_formatted_knowledge(category_ids)

        return formatted_knowledge

    except Exception:
        return ""
