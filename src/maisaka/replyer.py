"""
MaiSaka reply helper.
"""

from datetime import datetime
from typing import Any, Optional

from src.config.config import global_config

from .llm_service import MaiSakaLLMService

VISIBLE_REPLY_PREFIX = "\u3010\u9ea6\u9ea6\u7684\u53d1\u8a00\u3011"


def _normalize_content(content: str, limit: int = 500) -> str:
    normalized = " ".join((content or "").split())
    if len(normalized) > limit:
        return normalized[:limit] + "..."
    return normalized


def _format_message_time(_: dict[str, Any]) -> str:
    return datetime.now().strftime("%H:%M:%S")


def _extract_visible_assistant_reply(message: dict[str, Any]) -> str:
    if message.get("_type") == "perception":
        return ""

    content = (message.get("content", "") or "").strip()
    if not content:
        return ""

    marker = "[generated_reply]"
    if marker in content:
        _, visible_reply = content.rsplit(marker, 1)
        return _normalize_content(visible_reply)

    return ""


def _extract_guided_bot_reply(message: dict[str, Any]) -> str:
    content = (message.get("content", "") or "").strip()
    if content.startswith(VISIBLE_REPLY_PREFIX):
        return _normalize_content(content[len(VISIBLE_REPLY_PREFIX) :].strip())
    return ""


def format_chat_history(messages: list[dict[str, Any]]) -> str:
    """Format visible chat history for reply generation."""
    bot_nickname = global_config.bot.nickname.strip() or "Bot"
    parts: list[str] = []

    for message in messages:
        role = message.get("role", "")
        timestamp = _format_message_time(message)

        if role == "user":
            guided_reply = _extract_guided_bot_reply(message)
            if guided_reply:
                parts.append(f"{timestamp} {bot_nickname}（分析器指导的麦麦发言）：{guided_reply}")
                continue

            content = _normalize_content(message.get("content", "") or "")
            if content:
                parts.append(f"{timestamp} 用户：{content}")
            continue

        if role == "assistant":
            visible_reply = _extract_visible_assistant_reply(message)
            if visible_reply:
                parts.append(f"{timestamp} {bot_nickname}（你）：{visible_reply}")

    return "\n".join(parts)


class Replyer:
    """Generate visible replies from thoughts and context."""

    def __init__(self, llm_service: Optional[MaiSakaLLMService] = None):
        self._llm_service = llm_service
        self._enabled = True

    def set_llm_service(self, llm_service: MaiSakaLLMService) -> None:
        self._llm_service = llm_service

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    async def reply(self, reason: str, chat_history: list[dict[str, Any]]) -> str:
        if not self._enabled or not reason or self._llm_service is None:
            return "..."

        return await self._llm_service.generate_reply(reason, chat_history)
