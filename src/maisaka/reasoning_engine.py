"""Maisaka 推理引擎。"""

import asyncio
import time
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from src.chat.heart_flow.heartFC_utils import CycleDetail
from src.chat.message_receive.message import SessionMessage
from src.chat.replyer.replyer_manager import replyer_manager
from src.common.data_models.mai_message_data_model import UserInfo
from src.common.data_models.message_component_data_model import MessageSequence
from src.common.logger import get_logger
from src.config.config import global_config
from src.llm_models.payload_content.tool_option import ToolCall
from src.services import send_service

from .message_adapter import (
    build_message,
    build_visible_text_from_sequence,
    clone_message_sequence,
    format_speaker_content,
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
                            planner_started_at = time.time()
                            response = await self._runtime._chat_loop_service.chat_loop_step(self._runtime._chat_history)
                            cycle_detail.time_records["planner"] = time.time() - planner_started_at

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
            user_sequence = await self._build_message_sequence(message)
            visible_text = build_visible_text_from_sequence(user_sequence).strip()
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
            self._runtime._chat_history.append(history_message)
            self._trim_chat_history()

    async def _build_message_sequence(self, message: SessionMessage) -> MessageSequence:
        message_sequence = MessageSequence([])
        user_info = message.message_info.user_info
        speaker_name = user_info.user_cardname or user_info.user_nickname or user_info.user_id
        message_sequence.text(format_speaker_content(speaker_name, "", message.timestamp, message.message_id))

        appended_component = False
        if global_config.maisaka.direct_image_input:
            source_sequence = getattr(message, "maisaka_original_raw_message", message.raw_message)
        else:
            source_sequence = message.raw_message

        for component in clone_message_sequence(source_sequence).components:
            message_sequence.components.append(component)
            appended_component = True

        if not appended_component:
            if not message.processed_plain_text:
                await message.process()
            content = (message.processed_plain_text or "").strip()
            if content:
                message_sequence.text(content)

        return message_sequence

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
        user_message_count = sum(1 for message in self._runtime._chat_history if get_message_role(message) == "user")
        if user_message_count <= self._runtime._max_context_size:
            return

        trimmed_history = list(self._runtime._chat_history)
        removed_count = 0

        while user_message_count >= self._runtime._max_context_size and trimmed_history:
            removed_message = trimmed_history.pop(0)
            removed_count += 1
            if get_message_role(removed_message) == "user":
                user_message_count -= 1

        self._runtime._chat_history = trimmed_history
        self._runtime._log_history_trimmed(removed_count, user_message_count)

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

            if self._runtime._mcp_manager and self._runtime._mcp_manager.is_mcp_tool(tool_call.func_name):
                await handle_mcp_tool(tool_call, self._runtime._chat_history, self._runtime._mcp_manager)
                continue

            await handle_unknown_tool(tool_call, self._runtime._chat_history)

        return False

    async def _handle_reply(
        self,
        tool_call: ToolCall,
        latest_thought: str,
        anchor_message: SessionMessage,
    ) -> bool:
        tool_args = tool_call.args or {}
        target_message_id = str(tool_args.get("msg_id") or "").strip()
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
            f"target_msg_id={target_message_id} latest_thought={latest_thought!r}"
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

        try:
            success, reply_result = await replyer.generate_reply_with_context(
                reply_reason=latest_thought,
                stream_id=self._runtime.session_id,
                reply_message=target_message,
                chat_history=self._runtime._chat_history,
                log_reply=False,
            )
        except Exception:
            logger.exception(f"{self._runtime.log_prefix} reply generator crashed: target_msg_id={target_message_id}")
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
            f"target_msg_id={target_message_id} reply_text={reply_text!r}"
        )
        try:
            sent = await send_service.text_to_stream(
                text=reply_text,
                stream_id=self._runtime.session_id,
                set_reply=True,
                reply_message=target_message,
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

        bot_name = global_config.bot.nickname.strip() or "MaiSaka"
        self._runtime._chat_history.append(
            build_message(
                role="user",
                content=format_speaker_content(bot_name, reply_text, datetime.now()),
                source="guided_reply",
                platform=target_message.platform or anchor_message.platform,
                session_id=self._runtime.session_id,
                group_info=self._runtime._build_group_info(target_message),
                user_info=self._runtime._build_runtime_user_info(),
            )
        )
        return True

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
