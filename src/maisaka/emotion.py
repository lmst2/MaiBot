"""
MaiSaka - Emotion 模块
情绪感知分析，分析用户的情绪状态和言语态度。

注意：emotion.prompt 已迁移至主项目 prompts/ 目录
使用 prompt_manager.get_prompt("maidairy_emotion") 加载。
"""

from typing import List, Optional

from src.common.data_models.mai_message_data_model import MaiMessage

from .config import USER_NAME
from .message_adapter import get_message_role, get_message_text


def extract_user_messages(chat_history: List[MaiMessage], limit: Optional[int] = None) -> List[MaiMessage]:
    """
    从对话历史中提取用户消息。

    Args:
        chat_history: 完整的对话历史
        limit: 最多提取多少条用户消息，None 表示不限制

    Returns:
        只包含用户消息的列表
    """
    user_messages = [msg for msg in chat_history if get_message_role(msg) == "user"]
    if limit and len(user_messages) > limit:
        return user_messages[-limit:]
    return user_messages


def build_emotion_context(chat_history: List[MaiMessage]) -> str:
    """
    构建用于情绪分析的对话上下文文本。

    Args:
        chat_history: 完整的对话历史

    Returns:
        格式化后的对话上下文文本
    """
    # 获取最近的对话（约 8-10 条消息）
    recent_messages = chat_history[-10:] if len(chat_history) > 10 else chat_history

    context_parts = []
    for msg in recent_messages:
        role = get_message_role(msg)
        content = get_message_text(msg)

        if role == "user":
            context_parts.append(f"{USER_NAME}: {content}")
        elif role == "assistant":
            # 只显示 assistant 的实际发言，跳过感知信息
            if "【AI 感知】" not in content:
                context_parts.append(f"助手: {content}")

    return "\n".join(context_parts)
