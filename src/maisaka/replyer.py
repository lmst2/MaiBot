"""
MaiSaka - Reply 回复生成器
根据想法和上下文生成口语化回复。
"""

from typing import Optional
from llm_service import MaiSakaLLMService


def format_chat_history(messages: list) -> str:
    """将聊天消息列表格式化为可读文本。"""
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


class Replyer:
    """
    回复生成器。

    根据给定的想法（reason）和对话上下文，生成符合人设的口语化回复。
    """

    def __init__(self, llm_service: Optional[MaiSakaLLMService] = None):
        """
        初始化回复器。

        Args:
            llm_service: LLM 服务实例，如果为 None 则需要在调用前设置
        """
        self._llm_service = llm_service
        self._enabled = True

    def set_llm_service(self, llm_service: MaiSakaLLMService) -> None:
        """设置 LLM 服务"""
        self._llm_service = llm_service

    def set_enabled(self, enabled: bool) -> None:
        """启用/禁用回复功能"""
        self._enabled = enabled

    async def reply(self, reason: str, chat_history: list) -> str:
        """
        根据想法和上下文生成回复。

        Args:
            reason: 想要回复的方式、想法、内容（不包含具体回复内容）
            chat_history: 对话历史上下文

        Returns:
            生成的回复内容，失败时返回默认回复
        """
        if not self._enabled or not reason or self._llm_service is None:
            return "..."

        # 直接使用 LLM 服务的 generate_reply 方法
        # 该方法使用主项目的 replyer 模型配置
        return await self._llm_service.generate_reply(reason, chat_history)
