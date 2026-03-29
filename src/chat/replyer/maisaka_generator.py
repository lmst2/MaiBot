from datetime import datetime
from typing import Dict, List, Optional, Tuple

import random
import time

from src.chat.message_receive.chat_manager import BotChatSession
from src.chat.message_receive.message import SessionMessage
from src.common.data_models.reply_generation_data_models import (
    GenerationMetrics,
    LLMCompletionResult,
    ReplyGenerationResult,
)
from src.common.logger import get_logger
from src.common.prompt_i18n import load_prompt
from src.config.config import global_config
from src.core.types import ActionInfo
from src.services.llm_service import LLMServiceClient

from src.maisaka.message_adapter import (
    get_message_kind,
    get_message_role,
    get_message_source,
    get_message_text,
    parse_speaker_content,
)

logger = get_logger("maisaka_replyer")


class MaisakaReplyGenerator:
    """生成 Maisaka 的最终可见回复。"""

    def __init__(
        self,
        chat_stream: Optional[BotChatSession] = None,
        request_type: str = "maisaka_replyer",
    ) -> None:
        self.chat_stream = chat_stream
        self.request_type = request_type
        self.express_model = LLMServiceClient(
            task_name="replyer",
            request_type=request_type,
        )
        self._personality_prompt = self._build_personality_prompt()

    def _build_personality_prompt(self) -> str:
        """构建 replyer 使用的人设描述。"""
        try:
            bot_name = global_config.bot.nickname
            alias_names = global_config.bot.alias_names
            bot_aliases = f"，也有人叫你{','.join(alias_names)}" if alias_names else ""

            prompt_personality = global_config.personality.personality
            if (
                hasattr(global_config.personality, "states")
                and global_config.personality.states
                and hasattr(global_config.personality, "state_probability")
                and global_config.personality.state_probability > 0
                and random.random() < global_config.personality.state_probability
            ):
                prompt_personality = random.choice(global_config.personality.states)

            return f"你的名字是{bot_name}{bot_aliases}，你{prompt_personality};"
        except Exception as exc:
            logger.warning(f"Failed to build Maisaka personality prompt: {exc}")
            return "你的名字是麦麦，你是一个活泼可爱的 AI 助手。"

    @staticmethod
    def _normalize_content(content: str, limit: int = 500) -> str:
        normalized = " ".join((content or "").split())
        if len(normalized) > limit:
            return normalized[:limit] + "..."
        return normalized

    @staticmethod
    def _format_message_time(message: SessionMessage) -> str:
        return message.timestamp.strftime("%H:%M:%S")

    @staticmethod
    def _extract_visible_assistant_reply(message: SessionMessage) -> str:
        del message
        return ""

    def _extract_guided_bot_reply(self, message: SessionMessage) -> str:
        speaker_name, body = parse_speaker_content(get_message_text(message).strip())
        bot_nickname = global_config.bot.nickname.strip() or "Bot"
        if speaker_name == bot_nickname:
            return self._normalize_content(body.strip())
        return ""

    @staticmethod
    def _split_user_message_segments(raw_content: str) -> List[tuple[Optional[str], str]]:
        """按说话人拆分用户消息。"""
        segments: List[tuple[Optional[str], str]] = []
        current_speaker: Optional[str] = None
        current_lines: List[str] = []

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

    def _format_chat_history(self, messages: List[SessionMessage]) -> str:
        """格式化 replyer 使用的可见聊天记录。"""
        bot_nickname = global_config.bot.nickname.strip() or "Bot"
        parts: List[str] = []

        for message in messages:
            role = get_message_role(message)
            timestamp = self._format_message_time(message)

            if get_message_source(message) == "user_reference":
                continue

            if role == "user":
                guided_reply = self._extract_guided_bot_reply(message)
                if guided_reply:
                    parts.append(f"{timestamp} {bot_nickname}(you): {guided_reply}")
                    continue

                raw_content = get_message_text(message)
                for speaker_name, content_body in self._split_user_message_segments(raw_content):
                    content = self._normalize_content(content_body)
                    if not content:
                        continue
                    visible_speaker = speaker_name or global_config.maisaka.user_name.strip() or "User"
                    parts.append(f"{timestamp} {visible_speaker}: {content}")
                continue

            if role == "assistant":
                visible_reply = self._extract_visible_assistant_reply(message)
                if visible_reply:
                    parts.append(f"{timestamp} {bot_nickname}(you): {visible_reply}")

        return "\n".join(parts)

    def _build_prompt(
        self,
        chat_history: List[SessionMessage],
        reply_reason: str,
        expression_habits: str = "",
    ) -> str:
        """构建 Maisaka replyer 提示词。"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_history = self._format_chat_history(chat_history)

        try:
            system_prompt = load_prompt(
                "maidairy_replyer",
                bot_name=global_config.bot.nickname,
                identity=self._personality_prompt,
                reply_style=global_config.personality.reply_style,
            )
        except Exception:
            system_prompt = "你是一个友好的 AI 助手，请根据聊天记录自然回复。"

        extra_sections: List[str] = []
        if expression_habits.strip():
            extra_sections.append(expression_habits.strip())

        user_sections = [
            f"当前时间：{current_time}",
            f"【聊天记录】\n{formatted_history}",
        ]
        if extra_sections:
            user_sections.append("\n\n".join(extra_sections))
        user_sections.append(f"【你的想法】\n{reply_reason}")
        user_sections.append("现在，你说：")

        user_prompt = "\n\n".join(user_sections)
        return f"System: {system_prompt}\n\nUser: {user_prompt}"

    async def generate_reply_with_context(
        self,
        extra_info: str = "",
        reply_reason: str = "",
        available_actions: Optional[Dict[str, ActionInfo]] = None,
        chosen_actions: Optional[List[object]] = None,
        enable_tool: bool = True,
        from_plugin: bool = True,
        stream_id: Optional[str] = None,
        reply_message: Optional[SessionMessage] = None,
        reply_time_point: Optional[float] = None,
        think_level: int = 1,
        unknown_words: Optional[List[str]] = None,
        log_reply: bool = True,
        chat_history: Optional[List[SessionMessage]] = None,
        expression_habits: str = "",
        selected_expression_ids: Optional[List[int]] = None,
    ) -> Tuple[bool, ReplyGenerationResult]:
        """结合上下文生成 Maisaka 的最终可见回复。"""
        del available_actions
        del chosen_actions
        del enable_tool
        del extra_info
        del from_plugin
        del log_reply
        del reply_time_point
        del think_level
        del unknown_words

        result = ReplyGenerationResult()
        result.selected_expression_ids = list(selected_expression_ids or [])

        if chat_history is None:
            result.error_message = "chat_history is empty"
            return False, result

        logger.info(
            f"Maisaka replyer start: stream_id={stream_id} reply_reason={reply_reason!r} "
            f"history_size={len(chat_history)} target_message_id="
            f"{reply_message.message_id if reply_message else None} "
            f"expression_count={len(result.selected_expression_ids)}"
        )

        filtered_history = [
            message
            for message in chat_history
            if get_message_role(message) != "system"
            and get_message_kind(message) != "perception"
            and get_message_source(message) != "user_reference"
        ]
        prompt = self._build_prompt(
            chat_history=filtered_history,
            reply_reason=reply_reason or "",
            expression_habits=expression_habits,
        )
        result.completion.request_prompt = prompt

        if global_config.debug.show_replyer_prompt:
            logger.info(f"\nMaisaka replyer prompt:\n{prompt}\n")

        started_at = time.perf_counter()
        try:
            generation_result = await self.express_model.generate_response(prompt)
        except Exception as exc:
            logger.exception("Maisaka replyer call failed")
            result.error_message = str(exc)
            result.metrics = GenerationMetrics(
                overall_ms=round((time.perf_counter() - started_at) * 1000, 2),
            )
            return False, result

        response_text = (generation_result.response or "").strip()
        result.success = bool(response_text)
        result.completion = LLMCompletionResult(
            request_prompt=prompt,
            response_text=response_text,
            reasoning_text=generation_result.reasoning or "",
            model_name=generation_result.model_name or "",
            tool_calls=generation_result.tool_calls or [],
        )
        result.metrics = GenerationMetrics(
            overall_ms=round((time.perf_counter() - started_at) * 1000, 2),
        )

        if global_config.debug.show_replyer_reasoning and result.completion.reasoning_text:
            logger.info(f"Maisaka replyer reasoning:\n{result.completion.reasoning_text}")

        if not result.success:
            result.error_message = "replyer returned empty content"
            logger.warning("Maisaka replyer returned empty content")
            return False, result

        logger.info(
            f"Maisaka replyer success: response_text={response_text!r} "
            f"overall_ms={result.metrics.overall_ms} "
            f"selected_expression_ids={result.selected_expression_ids!r}"
        )
        result.text_fragments = [response_text]
        return True, result
