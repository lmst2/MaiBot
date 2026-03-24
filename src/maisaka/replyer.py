"""
MaiSaka reply helper.
"""

from typing import Optional

from src.common.data_models.mai_message_data_model import MaiMessage
from src.config.config import global_config

from .config import USER_NAME
from .llm_service import MaiSakaLLMService
from .message_adapter import get_message_role, get_message_text, is_perception_message, parse_speaker_content


def _normalize_content(content: str, limit: int = 500) -> str:
    normalized = " ".join((content or "").split())
    if len(normalized) > limit:
        return normalized[:limit] + "..."
    return normalized


def _format_message_time(message: MaiMessage) -> str:
    return message.timestamp.strftime("%H:%M:%S")


def _extract_visible_assistant_reply(message: MaiMessage) -> str:
    if is_perception_message(message):
        return ""
    return ""


def _extract_guided_bot_reply(message: MaiMessage) -> str:
    speaker_name, body = parse_speaker_content(get_message_text(message).strip())
    bot_nickname = global_config.bot.nickname.strip() or "Bot"
    if speaker_name == bot_nickname:
        return _normalize_content(body.strip())
    return ""


def format_chat_history(messages: list[MaiMessage]) -> str:
    """Format visible chat history for reply generation."""
    bot_nickname = global_config.bot.nickname.strip() or "Bot"
    parts: list[str] = []

    for message in messages:
        role = get_message_role(message)
        timestamp = _format_message_time(message)

        if role == "user":
            guided_reply = _extract_guided_bot_reply(message)
            if guided_reply:
                parts.append(f"{timestamp} {bot_nickname}(you): {guided_reply}")
                continue

            _, content_body = parse_speaker_content(get_message_text(message))
            content = _normalize_content(content_body)
            if content:
                parts.append(f"{timestamp} {USER_NAME}: {content}")
            continue

        if role == "assistant":
            visible_reply = _extract_visible_assistant_reply(message)
            if visible_reply:
                parts.append(f"{timestamp} {bot_nickname}(you): {visible_reply}")

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

    async def reply(self, reason: str, chat_history: list[MaiMessage]) -> str:
        if not self._enabled or not reason or self._llm_service is None:
            return "..."

        return await self._llm_service.generate_reply(reason, chat_history)
