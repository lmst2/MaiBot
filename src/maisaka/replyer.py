"""
MaiSaka reply helper.
"""

from typing import Optional

from src.chat.message_receive.message import SessionMessage
from src.config.config import global_config

from .llm_service import MaiSakaLLMService
from .message_adapter import get_message_role, get_message_text, is_perception_message, parse_speaker_content


def _normalize_content(content: str, limit: int = 500) -> str:
    normalized = " ".join((content or "").split())
    if len(normalized) > limit:
        return normalized[:limit] + "..."
    return normalized


def _format_message_time(message: SessionMessage) -> str:
    return message.timestamp.strftime("%H:%M:%S")


def _extract_visible_assistant_reply(message: SessionMessage) -> str:
    if is_perception_message(message):
        return ""
    return ""


def _extract_guided_bot_reply(message: SessionMessage) -> str:
    speaker_name, body = parse_speaker_content(get_message_text(message).strip())
    bot_nickname = global_config.bot.nickname.strip() or "Bot"
    if speaker_name == bot_nickname:
        return _normalize_content(body.strip())
    return ""


def _split_user_message_segments(raw_content: str) -> list[tuple[Optional[str], str]]:
    """Split a user message into speaker-labeled segments.

    A new segment only starts when a line explicitly begins with `[speaker]`.
    Continuation lines remain part of the current speaker's message.
    """
    segments: list[tuple[Optional[str], str]] = []
    current_speaker: Optional[str] = None
    current_lines: list[str] = []

    for raw_line in raw_content.splitlines():
        speaker_name, content_body = parse_speaker_content(raw_line)
        if speaker_name is not None:
            if current_lines:
                segments.append((current_speaker, "\n".join(current_lines)))
            current_speaker = speaker_name
            current_lines = [content_body]
            continue

        current_lines.append(raw_line)

    if current_lines:
        segments.append((current_speaker, "\n".join(current_lines)))

    return segments


def format_chat_history(messages: list[SessionMessage]) -> str:
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

            raw_content = get_message_text(message)
            for speaker_name, content_body in _split_user_message_segments(raw_content):
                content = _normalize_content(content_body)
                if not content:
                    continue
                visible_speaker = speaker_name or global_config.maisaka.user_name.strip() or "用户"
                parts.append(f"{timestamp} {visible_speaker}: {content}")
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

    async def reply(self, reason: str, chat_history: list[SessionMessage]) -> str:
        if not self._enabled or not reason or self._llm_service is None:
            return "..."

        return await self._llm_service.generate_reply(reason, chat_history)
