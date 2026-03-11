"""
MaiSaka - LLM 服务工具函数
"""

from typing import Literal

# ──────────────────── 消息类型 ────────────────────

MessageType = Literal["user", "assistant", "system", "perception"]

# 内部使用的字段前缀，用于标记不应发送给 API 的元数据
INTERNAL_FIELD_PREFIX = "_"

# 消息类型字段名
MSG_TYPE_FIELD = "_type"


# ──────────────────── 消息构建 ────────────────────

def build_message(role: str, content: str, msg_type: MessageType = "user", **kwargs) -> dict:
    """
    构建消息字典，包含消息类型标记。

    Args:
        role: 消息角色 (user/assistant/system)
        content: 消息内容
        msg_type: 消息类型 (user/assistant/system/perception)
        **kwargs: 其他字段（如 tool_calls）

    Returns:
        消息字典
    """
    msg = {"role": role, "content": content, MSG_TYPE_FIELD: msg_type, **kwargs}
    return msg


def filter_for_api(messages: list[dict]) -> list[dict]:
    """
    过滤消息列表，移除内部字段，用于发送给 API。

    Args:
        messages: 原始消息列表

    Returns:
        过滤后的消息列表（移除所有以 _ 开头的字段）
    """
    return [
        {k: v for k, v in msg.items() if not k.startswith(INTERNAL_FIELD_PREFIX)}
        for msg in messages
    ]


def filter_by_type(messages: list[dict], msg_type: MessageType) -> list[dict]:
    """
    按消息类型过滤消息列表。

    Args:
        messages: 原始消息列表
        msg_type: 要保留的消息类型

    Returns:
        只包含指定类型的消息列表
    """
    return [msg for msg in messages if msg.get(MSG_TYPE_FIELD) == msg_type]


def remove_last_perception(messages: list[dict]) -> None:
    """
    移除最后一条感知消息（直接修改原列表）。

    Args:
        messages: 消息列表（会被原地修改）
    """
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get(MSG_TYPE_FIELD) == "perception":
            messages.pop(i)
            break


def format_chat_history(messages: list) -> str:
    """将聊天消息列表格式化为可读文本，用于子代理上下文构建。"""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content", "") or ""
        if role == "system":
            parts.append(f"[系统] {content[:500]}")
        elif role == "user":
            parts.append(f"[用户] {content[:500]}")
        elif role == "assistant":
            if content:
                parts.append(f"[助手思考] {content[:500]}")
            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {})
                name = func.get("name", "?")
                args = func.get("arguments", "")
                if isinstance(args, str) and len(args) > 200:
                    args = args[:200] + "..."
                parts.append(f"[助手调用 {name}] {args}")
        elif role == "tool":
            parts.append(f"[工具结果] {content[:300]}")
    return "\n".join(parts)


def format_chat_history_for_eq(messages: list) -> str:
    """
    将聊天消息列表格式化为可读文本，专门用于情商模块。

    只包含三种内容：
    1. 模型自身思考内容（assistant 的 content）
    2. 模型 say 的结果内容（say 工具的结果）
    3. 用户回复内容（user 消息）

    不包含：工具调用本身、其他工具的结果
    """
    parts: list[str] = []
    say_tool_call_ids = set()

    # 第一遍：收集所有 say 工具的 tool_call_id
    for msg in messages:
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {})
                if func.get("name") == "say":
                    say_tool_call_ids.add(tc.get("id", ""))

    # 第二遍：格式化消息
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content", "") or ""

        if role == "user":
            parts.append(f"[用户] {content[:500]}")
        elif role == "assistant":
            # 只包含助手思考内容，不包含工具调用本身
            if content:
                parts.append(f"[助手思考] {content[:500]}")
        elif role == "tool":
            # 只包含 say 工具的结果
            tool_call_id = msg.get("tool_call_id", "")
            if tool_call_id in say_tool_call_ids:
                parts.append(f"[助手说] {content[:500]}")

    return "\n".join(parts)
