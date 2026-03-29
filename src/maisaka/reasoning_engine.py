"""Maisaka 推理引擎。"""

from datetime import datetime
from typing import TYPE_CHECKING, Optional

import asyncio
import difflib
import json
import re
import time

from sqlmodel import select

from src.chat.heart_flow.heartFC_utils import CycleDetail
from src.chat.message_receive.message import SessionMessage
from src.chat.replyer.replyer_manager import replyer_manager
from src.chat.utils.utils import get_bot_account
from src.common.database.database import get_db_session
from src.common.database.database_model import Jargon
from src.common.data_models.mai_message_data_model import UserInfo
from src.common.data_models.message_component_data_model import MessageSequence, TextComponent
from src.common.logger import get_logger
from src.config.config import global_config
from src.learners.jargon_explainer import search_jargon
from src.llm_models.payload_content.tool_option import ToolCall
from src.services import database_service as database_api, send_service

from .message_adapter import (
    build_message,
    build_visible_text_from_sequence,
    clone_message_sequence,
    format_speaker_content,
    get_message_source,
    get_message_text,
    get_message_role,
)
from .tool_handlers import (
    handle_mcp_tool,
    handle_unknown_tool,
)

if TYPE_CHECKING:
    from .runtime import MaisakaHeartFlowChatting

logger = get_logger("maisaka_reasoning_engine")


class MaisakaReasoningEngine:
    """负责内部思考、推理与工具执行。"""

    def __init__(self, runtime: "MaisakaHeartFlowChatting") -> None:
        self._runtime = runtime
        self._last_reasoning_content: str = ""
        self._shown_jargons: set[str] = set()  # 已在参考消息中展示过的 jargon

    async def run_loop(self) -> None:
        """独立消费消息批次，并执行对应的内部思考轮次。"""
        try:
            while self._runtime._running:
                cached_messages = await self._runtime._internal_turn_queue.get()
                if not cached_messages:
                    self._runtime._internal_turn_queue.task_done()
                    continue

                self._runtime._agent_state = self._runtime._STATE_RUNNING
                await self._ingest_messages(cached_messages)

                anchor_message = cached_messages[-1]
                try:
                    for round_index in range(self._runtime._max_internal_rounds):
                        cycle_detail = self._start_cycle()
                        self._runtime._log_cycle_started(cycle_detail, round_index)
                        try:
                            # 每次LLM生成前，动态添加参考消息到最新位置
                            reference_added = self._append_jargon_reference_message()
                            planner_started_at = time.time()
                            response = await self._runtime._chat_loop_service.chat_loop_step(self._runtime._chat_history)
                            cycle_detail.time_records["planner"] = time.time() - planner_started_at

                            # LLM调用后，移除刚才添加的参考消息（一次性使用）
                            if reference_added and self._runtime._chat_history:
                                # 从末尾往前查找并移除参考消息
                                for i in range(len(self._runtime._chat_history) - 1, -1, -1):
                                    if get_message_source(self._runtime._chat_history[i]) == "user_reference":
                                        self._runtime._chat_history.pop(i)
                                        break

                            reasoning_content = response.content or ""
                            if self._should_replace_reasoning(reasoning_content):
                                response.content = "让我根据新情况重新思考："
                                response.raw_message.content = "让我根据新情况重新思考："
                                logger.info(f"{self._runtime.log_prefix} reasoning content replaced due to high similarity")

                            self._last_reasoning_content = reasoning_content
                            response.raw_message.platform = anchor_message.platform
                            response.raw_message.session_id = self._runtime.session_id
                            response.raw_message.message_info.group_info = self._runtime._build_group_info(anchor_message)
                            self._runtime._chat_history.append(response.raw_message)

                            if response.tool_calls:
                                tool_started_at = time.time()
                                should_pause = await self._handle_tool_calls(
                                    response.tool_calls,
                                    response.content or "",
                                    anchor_message,
                                )
                                cycle_detail.time_records["tool_calls"] = time.time() - tool_started_at
                                if should_pause:
                                    break
                                continue

                            if response.content:
                                continue

                            break
                        finally:
                            self._end_cycle(cycle_detail)
                finally:
                    if self._runtime._agent_state == self._runtime._STATE_RUNNING:
                        self._runtime._agent_state = self._runtime._STATE_STOP
                    self._runtime._internal_turn_queue.task_done()
        except asyncio.CancelledError:
            self._runtime._log_internal_loop_cancelled()
            raise
        except Exception:
            logger.exception("%s Maisaka internal loop crashed", self._runtime.log_prefix)
            raise

    async def _ingest_messages(self, messages: list[SessionMessage]) -> None:
        """处理传入消息列表，将其转换为历史消息并加入聊天历史缓存。"""
        for message in messages:
            # 构建用户消息序列
            user_sequence, visible_text = await self._build_message_sequence(message)
            if not user_sequence.components:
                continue

            history_message = build_message(
                role="user",
                content=visible_text,
                source="user",
                timestamp=message.timestamp,
                platform=message.platform,
                session_id=self._runtime.session_id,
                group_info=self._runtime._build_group_info(message),
                user_info=self._runtime._build_runtime_user_info(),
                raw_message=user_sequence,
                display_text=visible_text,
            )
            self._insert_chat_history_message(history_message)
            self._trim_chat_history()

    async def _build_message_sequence(self, message: SessionMessage) -> tuple[MessageSequence, str]:
        message_sequence = MessageSequence([])
        planner_prefix = self._build_planner_user_prefix(message)

        appended_component = False
        if global_config.maisaka.direct_image_input:
            source_sequence = getattr(message, "maisaka_original_raw_message", message.raw_message)
        else:
            source_sequence = message.raw_message

        planner_components = clone_message_sequence(source_sequence).components
        if planner_components and isinstance(planner_components[0], TextComponent):
            planner_components[0].text = planner_prefix + planner_components[0].text
        else:
            planner_components.insert(0, TextComponent(planner_prefix))

        for component in planner_components:
            message_sequence.components.append(component)
            appended_component = True

        legacy_visible_text = self._build_legacy_visible_text(message, source_sequence)
        if not appended_component:
            if not message.processed_plain_text:
                await message.process()
            content = (message.processed_plain_text or "").strip()
            if content:
                message_sequence.text(planner_prefix + content)
                legacy_visible_text = self._build_legacy_visible_text_from_text(message, content)

        return message_sequence, legacy_visible_text

    @staticmethod
    def _build_planner_user_prefix(message: SessionMessage) -> str:
        user_info = message.message_info.user_info
        timestamp_text = message.timestamp.strftime("%H:%M:%S")
        user_name = user_info.user_nickname or user_info.user_id
        group_card = user_info.user_cardname or ""
        message_id = message.message_id or ""
        return (
            f"[时间]{timestamp_text}\n"
            f"[用户]{user_name}\n"
            f"[用户群昵称]{group_card}\n"
            f"[msg_id]{message_id}\n"
            "[发言内容]"
        )

    def _build_legacy_visible_text(self, message: SessionMessage, source_sequence: MessageSequence) -> str:
        user_info = message.message_info.user_info
        speaker_name = user_info.user_cardname or user_info.user_nickname or user_info.user_id
        legacy_sequence = MessageSequence([])
        legacy_sequence.text(format_speaker_content(speaker_name, "", message.timestamp, message.message_id))
        for component in clone_message_sequence(source_sequence).components:
            legacy_sequence.components.append(component)
        return build_visible_text_from_sequence(legacy_sequence).strip()

    def _build_legacy_visible_text_from_text(self, message: SessionMessage, content: str) -> str:
        user_info = message.message_info.user_info
        speaker_name = user_info.user_cardname or user_info.user_nickname or user_info.user_id
        return format_speaker_content(speaker_name, content, message.timestamp, message.message_id).strip()

    def _insert_chat_history_message(self, message: SessionMessage) -> int:
        """按时间顺序将消息插入聊天历史，同时保留 system 消息在最前。"""
        if not self._runtime._chat_history:
            self._runtime._chat_history.append(message)
            return 0

        insert_at = len(self._runtime._chat_history)
        for index, existing_message in enumerate(self._runtime._chat_history):
            if get_message_role(existing_message) == "system":
                continue
            if existing_message.timestamp > message.timestamp:
                insert_at = index
                break

        self._runtime._chat_history.insert(insert_at, message)
        return insert_at

    def _append_jargon_reference_message(self) -> bool:
        """每次LLM生成前，如果命中了黑话词条，则添加一条参考信息消息到聊天历史末尾。

        Returns:
            bool: 是否添加了参考消息
        """
        content = self._build_user_history_corpus()
        if not content:
            return False

        matched_words = self._find_jargon_words_in_text(content)
        if not matched_words:
            return False

        # 记录已展示的 jargon
        for word in matched_words:
            self._shown_jargons.add(word.lower())

        reference_text = (
            "[参考信息]\n"
            f"{','.join(matched_words)}可能是jargon，可以使用query_jargon来查看其含义"
        )
        reference_sequence = MessageSequence([TextComponent(reference_text)])

        # 使用当前时间作为时间戳
        reference_message = build_message(
            role="user",
            content="",
            source="user_reference",
            timestamp=datetime.now(),
            platform=self._runtime.chat_stream.platform,
            session_id=self._runtime.session_id,
            group_info=self._runtime._build_group_info(),
            user_info=self._runtime._build_runtime_user_info(),
            raw_message=reference_sequence,
            display_text=reference_text,
        )
        self._runtime._chat_history.append(reference_message)
        return True

    def _build_user_history_corpus(self) -> str:
        """拼接当前聊天记录内所有用户消息的正文，用于统一匹配黑话。"""
        parts: list[str] = []
        for history_message in self._runtime._chat_history:
            if get_message_role(history_message) != "user":
                continue
            if get_message_source(history_message) != "user":
                continue
            text = (get_message_text(history_message) or "").strip()
            if not text:
                continue
            parts.append(text)

        return "\n".join(parts)

    def _find_jargon_words_in_text(self, content: str) -> list[str]:
        """匹配正文中出现的 jargon 词条。"""
        lowered_content = content.lower()
        matched_entries: list[tuple[int, int, int, str]] = []
        seen_words: set[str] = set()

        with get_db_session(auto_commit=False) as session:
            query = (
                select(Jargon)
                .where(Jargon.is_jargon.is_(True))
                .order_by(Jargon.count.desc())  # type: ignore[attr-defined]
            )
            jargons = session.exec(query).all()

            for jargon in jargons:
                jargon_content = str(jargon.content or "").strip()
                if not jargon_content:
                    continue
                # meaning 为空的不匹配
                if not str(jargon.meaning or "").strip():
                    continue
                normalized_content = jargon_content.lower()
                if normalized_content in seen_words:
                    continue
                # 跳过已经展示过的 jargon
                if normalized_content in self._shown_jargons:
                    continue
                if not self._is_visible_jargon(jargon):
                    continue
                match_position = self._get_jargon_match_position(jargon_content, lowered_content, content)
                if match_position is None:
                    continue

                seen_words.add(normalized_content)
                matched_entries.append((match_position, -len(jargon_content), -int(jargon.count or 0), jargon_content))

        matched_entries.sort()
        return [matched_content for _, _, _, matched_content in matched_entries[:8]]

    def _is_visible_jargon(self, jargon: Jargon) -> bool:
        """判断当前会话是否可见该 jargon。"""
        if global_config.expression.all_global_jargon or bool(jargon.is_global):
            return True

        try:
            session_id_dict = json.loads(jargon.session_id_dict or "{}")
        except (TypeError, json.JSONDecodeError):
            logger.warning(f"Failed to parse jargon.session_id_dict: jargon_id={jargon.id}")
            return False
        return self._runtime.session_id in session_id_dict

    @staticmethod
    def _get_jargon_match_position(jargon_content: str, lowered_content: str, original_content: str) -> Optional[int]:
        """返回 jargon 在文本中的首次命中位置，未命中时返回 `None`。"""
        if re.search(r"[\u4e00-\u9fff]", jargon_content):
            match_index = original_content.lower().find(jargon_content.lower())
            return match_index if match_index >= 0 else None

        pattern = rf"\b{re.escape(jargon_content.lower())}\b"
        match = re.search(pattern, lowered_content)
        if match is None:
            return None
        return match.start()

    def _start_cycle(self) -> CycleDetail:
        """开始一轮 Maisaka 思考循环。"""
        self._runtime._cycle_counter += 1
        self._runtime._current_cycle_detail = CycleDetail(cycle_id=self._runtime._cycle_counter)
        self._runtime._current_cycle_detail.thinking_id = f"maisaka_tid{round(time.time(), 2)}"
        return self._runtime._current_cycle_detail

    def _end_cycle(self, cycle_detail: CycleDetail, only_long_execution: bool = True) -> CycleDetail:
        """结束并记录一轮 Maisaka 思考循环。"""
        cycle_detail.end_time = time.time()
        self._runtime.history_loop.append(cycle_detail)

        timer_strings = [
            f"{name}: {duration:.2f}s"
            for name, duration in cycle_detail.time_records.items()
            if not only_long_execution or duration >= 0.1
        ]
        self._runtime._log_cycle_completed(cycle_detail, timer_strings)
        return cycle_detail

    def _trim_chat_history(self) -> None:
        """裁剪聊天历史，保证用户消息数量不超过配置限制。"""
        counted_roles = {"user", "assistant"}
        conversation_message_count = sum(
            1 for message in self._runtime._chat_history if get_message_role(message) in counted_roles
        )
        if conversation_message_count <= self._runtime._max_context_size:
            return

        trimmed_history = list(self._runtime._chat_history)
        removed_count = 0

        while conversation_message_count >= self._runtime._max_context_size and trimmed_history:
            removed_message = trimmed_history.pop(0)
            removed_count += 1
            if get_message_role(removed_message) in counted_roles:
                conversation_message_count -= 1

        self._runtime._chat_history = trimmed_history
        self._runtime._log_history_trimmed(removed_count, conversation_message_count)

    @staticmethod
    def _calculate_similarity(text1: str, text2: str) -> float:
        """计算两个文本之间的相似度。

        Args:
            text1: 第一个文本
            text2: 第二个文本

        Returns:
            float: 相似度值，范围 0-1，1 表示完全相同
        """
        return difflib.SequenceMatcher(None, text1, text2).ratio()

    def _should_replace_reasoning(self, current_content: str) -> bool:
        """判断是否需要替换推理内容。

        当当前推理内容与上一次相似度大于90%时，返回True。

        Args:
            current_content: 当前的推理内容

        Returns:
            bool: 是否需要替换
        """
        if not self._last_reasoning_content or not current_content:
            return False

        similarity = self._calculate_similarity(current_content, self._last_reasoning_content)
        logger.info(f"{self._runtime.log_prefix} reasoning similarity: {similarity:.2f}")
        return similarity > 0.9

    async def _handle_tool_calls(
        self,
        tool_calls: list[ToolCall],
        latest_thought: str,
        anchor_message: SessionMessage,
    ) -> bool:
        for tool_call in tool_calls:
            if tool_call.func_name == "reply":
                reply_sent = await self._handle_reply(tool_call, latest_thought, anchor_message)
                if reply_sent:
                    return True
                continue

            if tool_call.func_name == "no_reply":
                self._runtime._chat_history.append(
                    self._build_tool_message(
                        tool_call,
                        "No visible reply was sent for this round.",
                    )
                )
                continue

            if tool_call.func_name == "query_jargon":
                await self._handle_query_jargon(tool_call)
                continue

            if tool_call.func_name == "wait":
                seconds = (tool_call.args or {}).get("seconds", 30)
                try:
                    wait_seconds = int(seconds)
                except (TypeError, ValueError):
                    wait_seconds = 30
                wait_seconds = max(0, wait_seconds)
                self._runtime._chat_history.append(
                    self._build_tool_message(
                        tool_call,
                        f"Waiting for future input for up to {wait_seconds} seconds.",
                    )
                )
                self._runtime._enter_wait_state(seconds=wait_seconds)
                return True

            if tool_call.func_name == "stop":
                self._runtime._chat_history.append(
                    self._build_tool_message(
                        tool_call,
                        "Conversation loop paused until a new message arrives.",
                    )
                )
                self._runtime._enter_stop_state()
                return True

            if tool_call.func_name == "send_emoji":
                await self._handle_send_emoji(tool_call, anchor_message)
                continue

            if self._runtime._mcp_manager and self._runtime._mcp_manager.is_mcp_tool(tool_call.func_name):
                await handle_mcp_tool(tool_call, self._runtime._chat_history, self._runtime._mcp_manager)
                continue

            await handle_unknown_tool(tool_call, self._runtime._chat_history)

        return False

    async def _handle_query_jargon(self, tool_call: ToolCall) -> None:
        tool_args = tool_call.args or {}
        raw_words = tool_args.get("words")

        if not isinstance(raw_words, list):
            self._runtime._chat_history.append(
                self._build_tool_message(tool_call, "query_jargon requires a words array.")
            )
            return

        words: list[str] = []
        seen_words: set[str] = set()
        for item in raw_words:
            if not isinstance(item, str):
                continue
            word = item.strip()
            if not word or word in seen_words:
                continue
            seen_words.add(word)
            words.append(word)

        if not words:
            self._runtime._chat_history.append(
                self._build_tool_message(tool_call, "query_jargon requires at least one non-empty word.")
            )
            return

        logger.info(f"{self._runtime.log_prefix} query_jargon triggered: words={words!r}")

        results: list[dict[str, object]] = []
        for word in words:
            exact_matches = search_jargon(
                keyword=word,
                chat_id=self._runtime.session_id,
                limit=5,
                case_sensitive=False,
                fuzzy=False,
            )
            matched_entries = exact_matches or search_jargon(
                keyword=word,
                chat_id=self._runtime.session_id,
                limit=5,
                case_sensitive=False,
                fuzzy=True,
            )

            results.append(
                {
                    "word": word,
                    "found": bool(matched_entries),
                    "matches": matched_entries,
                }
            )

        logger.info(f"{self._runtime.log_prefix} query_jargon finished: results={results!r}")
        self._runtime._chat_history.append(
            self._build_tool_message(
                tool_call,
                json.dumps({"results": results}, ensure_ascii=False),
            )
        )

    async def _handle_reply(
        self,
        tool_call: ToolCall,
        latest_thought: str,
        anchor_message: SessionMessage,
    ) -> bool:
        tool_args = tool_call.args or {}
        target_message_id = str(tool_args.get("msg_id") or "").strip()
        quote_reply = bool(tool_args.get("quote", True))
        raw_unknown_words = tool_args.get("unknown_words")
        unknown_words = raw_unknown_words if isinstance(raw_unknown_words, list) else None
        if not target_message_id:
            self._runtime._chat_history.append(
                self._build_tool_message(tool_call, "reply requires a valid msg_id argument.")
            )
            return False

        target_message = self._runtime._source_messages_by_id.get(target_message_id)
        if target_message is None:
            self._runtime._chat_history.append(
                self._build_tool_message(tool_call, f"reply target msg_id not found: {target_message_id}")
            )
            return False

        logger.info(
            f"{self._runtime.log_prefix} reply tool triggered: "
            f"target_msg_id={target_message_id} quote={quote_reply} latest_thought={latest_thought!r}"
        )
        logger.info(f"{self._runtime.log_prefix} acquiring Maisaka reply generator")
        try:
            replyer = replyer_manager.get_replyer(
                chat_stream=self._runtime.chat_stream,
                request_type="maisaka_replyer",
                replyer_type="maisaka",
            )
        except Exception:
            logger.exception(
                f"{self._runtime.log_prefix} replyer_manager.get_replyer crashed: "
                f"target_msg_id={target_message_id}"
            )
            self._runtime._chat_history.append(
                self._build_tool_message(tool_call, "Maisaka reply generator acquisition crashed.")
            )
            return False

        if replyer is None:
            logger.error(f"{self._runtime.log_prefix} failed to acquire Maisaka reply generator")
            self._runtime._chat_history.append(
                self._build_tool_message(tool_call, "Maisaka reply generator is unavailable.")
            )
            return False

        logger.info(f"{self._runtime.log_prefix} acquired Maisaka reply generator successfully")

        logger.info(f"{self._runtime.log_prefix} calling generate_reply_with_context: target_msg_id={target_message_id}")
        try:
            success, reply_result = await replyer.generate_reply_with_context(
                reply_reason=latest_thought,
                stream_id=self._runtime.session_id,
                reply_message=target_message,
                chat_history=self._runtime._chat_history,
                unknown_words=unknown_words,
                log_reply=False,
            )
        except Exception as exc:
            import traceback
            logger.error(
                f"{self._runtime.log_prefix} reply generator crashed: target_msg_id={target_message_id} "
                f"exc_type={type(exc).__name__} exc_msg={str(exc)}\n{traceback.format_exc()}"
            )
            self._runtime._chat_history.append(
                self._build_tool_message(tool_call, "Visible reply generation crashed.")
            )
            return False

        logger.info(
            f"{self._runtime.log_prefix} reply generator finished: "
            f"success={success} response_text={reply_result.completion.response_text!r} "
            f"error={reply_result.error_message!r}"
        )
        reply_text = reply_result.completion.response_text.strip() if success else ""
        if not reply_text:
            logger.warning(
                f"{self._runtime.log_prefix} reply generator returned empty text: "
                f"target_msg_id={target_message_id} error={reply_result.error_message!r}"
            )
            self._runtime._chat_history.append(
                self._build_tool_message(tool_call, "Visible reply generation failed.")
            )
            return False

        logger.info(
            f"{self._runtime.log_prefix} sending guided reply: "
            f"target_msg_id={target_message_id} quote={quote_reply} reply_text={reply_text!r}"
        )
        try:
            sent = await send_service.text_to_stream(
                text=reply_text,
                stream_id=self._runtime.session_id,
                set_reply=quote_reply,
                reply_message=target_message if quote_reply else None,
                selected_expressions=reply_result.selected_expression_ids or None,
                typing=False,
            )
        except Exception:
            logger.exception(
                f"{self._runtime.log_prefix} send_service.text_to_stream crashed "
                f"for target_msg_id={target_message_id}"
            )
            self._runtime._chat_history.append(
                self._build_tool_message(tool_call, "Visible reply send crashed.")
            )
            return False

        logger.info(
            f"{self._runtime.log_prefix} guided reply send result: "
            f"target_msg_id={target_message_id} sent={sent}"
        )
        tool_result = "Visible reply generated and sent." if sent else "Visible reply generation succeeded but send failed."
        self._runtime._chat_history.append(self._build_tool_message(tool_call, tool_result))
        if not sent:
            return False

        target_user_info = target_message.message_info.user_info
        target_user_name = (
            target_user_info.user_cardname
            or target_user_info.user_nickname
            or target_user_info.user_id
        )
        if self._runtime.chat_stream is not None:
            await database_api.store_tool_info(
                chat_stream=self._runtime.chat_stream,
                display_prompt=f"你对{target_user_name}进行了回复：{reply_text}",
                tool_data={
                    "msg_id": target_message_id,
                    "quote": quote_reply,
                    "reply_text": reply_text,
                },
                tool_name="reply",
                tool_reasoning=latest_thought,
            )

        target_platform = target_message.platform or anchor_message.platform
        bot_name = global_config.bot.nickname.strip() or "MaiSaka"
        bot_user_info = UserInfo(
            user_id=get_bot_account(target_platform) or "maisaka_assistant",
            user_nickname=bot_name,
            user_cardname=None,
        )
        history_message = build_message(
            role="assistant",
            content=reply_text,
            source="guided_reply",
            platform=target_platform,
            session_id=self._runtime.session_id,
            group_info=self._runtime._build_group_info(target_message),
            user_info=bot_user_info,
        )
        structured_visible_text = f"{self._build_planner_user_prefix(history_message)}{reply_text}"
        history_message.display_message = structured_visible_text
        history_message.processed_plain_text = structured_visible_text
        self._runtime._chat_history.append(history_message)
        return True

    async def _handle_send_emoji(self, tool_call: ToolCall, anchor_message: SessionMessage) -> None:
        """处理发送表情包的工具调用。

        Args:
            tool_call: 工具调用对象
            anchor_message: 锚点消息
        """
        from src.chat.emoji_system.emoji_manager import emoji_manager
        from src.common.utils.utils_image import ImageUtils
        import random

        tool_args = tool_call.args or {}
        emotion = str(tool_args.get("emotion") or "").strip()

        logger.info(f"{self._runtime.log_prefix} send_emoji tool triggered: emotion={emotion!r}")

        # 获取表情包列表
        if not emoji_manager.emojis:
            self._runtime._chat_history.append(
                self._build_tool_message(tool_call, "No emojis available in the emoji library.")
            )
            return

        # 根据情感选择表情包
        selected_emoji = None
        if emotion:
            # 尝试找到匹配情感的表情包
            matching_emojis = [
                emoji for emoji in emoji_manager.emojis
                if emotion.lower() in (e.lower() for e in emoji.emotion)
            ]
            if matching_emojis:
                selected_emoji = random.choice(matching_emojis)
                logger.info(
                    f"{self._runtime.log_prefix} found {len(matching_emojis)} emojis matching emotion '{emotion}', "
                    f"selected: {selected_emoji.description}"
                )

        # 如果没有找到匹配的情感表情包，随机选择一个
        if selected_emoji is None:
            selected_emoji = random.choice(emoji_manager.emojis)
            logger.info(
                f"{self._runtime.log_prefix} no emoji matched emotion '{emotion}', "
                f"randomly selected: {selected_emoji.description}"
            )

        # 更新表情包使用次数
        emoji_manager.update_emoji_usage(selected_emoji)

        # 获取表情包的 base64 数据
        try:
            emoji_base64 = ImageUtils.image_path_to_base64(str(selected_emoji.full_path))
            if not emoji_base64:
                raise ValueError("Failed to convert emoji image to base64")
        except Exception as exc:
            logger.error(
                f"{self._runtime.log_prefix} failed to convert emoji to base64: {exc}"
            )
            self._runtime._chat_history.append(
                self._build_tool_message(tool_call, f"Failed to send emoji: {exc}")
            )
            return

        # 发送表情包
        try:
            sent = await send_service.emoji_to_stream(
                emoji_base64=emoji_base64,
                stream_id=self._runtime.session_id,
                storage_message=True,
                set_reply=False,
                reply_message=None,
            )
        except Exception as exc:
            logger.exception(
                f"{self._runtime.log_prefix} send_service.emoji_to_stream crashed: {exc}"
            )
            self._runtime._chat_history.append(
                self._build_tool_message(tool_call, f"Emoji send crashed: {exc}")
            )
            return

        if sent:
            logger.info(
                f"{self._runtime.log_prefix} emoji sent successfully: "
                f"description={selected_emoji.description!r} emotion={selected_emoji.emotion}"
            )
            self._runtime._chat_history.append(
                self._build_tool_message(
                    tool_call,
                    f"Sent emoji: {selected_emoji.description} (emotion: {', '.join(selected_emoji.emotion)})"
                )
            )
        else:
            logger.warning(f"{self._runtime.log_prefix} emoji send failed")
            self._runtime._chat_history.append(
                self._build_tool_message(tool_call, "Failed to send emoji.")
            )

    def _build_tool_message(self, tool_call: ToolCall, content: str) -> SessionMessage:
        return build_message(
            role="tool",
            content=content,
            source="tool",
            tool_call_id=tool_call.call_id,
            platform=self._runtime.chat_stream.platform,
            session_id=self._runtime.session_id,
            group_info=self._runtime._build_group_info(),
            user_info=UserInfo(user_id="maisaka_tool", user_nickname="tool", user_cardname=None),
        )
