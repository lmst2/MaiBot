"""
Maisaka runtime for non-CLI integrations.
"""

import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from src.chat.heart_flow.heartFC_utils import CycleDetail
from src.chat.message_receive.chat_manager import BotChatSession, chat_manager
from src.chat.message_receive.message import SessionMessage
from src.common.data_models.mai_message_data_model import GroupInfo, UserInfo
from src.common.data_models.message_component_data_model import MessageSequence
from src.common.logger import get_logger
from src.config.config import global_config
from src.llm_models.payload_content.tool_option import ToolCall
from src.services import send_service

from .llm_service import MaiSakaLLMService
from .mcp_client import MCPManager
from .message_adapter import (
    build_message,
    build_visible_text_from_sequence,
    clone_message_sequence,
    format_speaker_content,
    get_message_role,
)
from .reasoning_engine import MaisakaReasoningEngine
from .tool_handlers import (
    handle_mcp_tool,
    handle_unknown_tool,
)

logger = get_logger("maisaka_runtime")


class MaisakaHeartFlowChatting:
    """Session-scoped Maisaka runtime that replaces the HFC planner and reply loop."""

    _STATE_RUNNING: Literal["running"] = "running"
    _STATE_WAIT: Literal["wait"] = "wait"
    _STATE_STOP: Literal["stop"] = "stop"

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.chat_stream: Optional[BotChatSession] = chat_manager.get_session_by_session_id(session_id)
        if self.chat_stream is None:
            raise ValueError(f"Session not found for Maisaka runtime: {session_id}")

        session_name = chat_manager.get_session_name(session_id) or session_id
        self.log_prefix = f"[{session_name}]"
        self._llm_service = MaiSakaLLMService(api_key="", base_url=None, model="")
        self._chat_history: list[SessionMessage] = []
        self.history_loop: list[CycleDetail] = []
        self.message_cache: list[SessionMessage] = []
        self._internal_turn_queue: asyncio.Queue[list[SessionMessage]] = asyncio.Queue()
        self._message_queue: asyncio.Queue[SessionMessage] = asyncio.Queue()
        self._mcp_manager: Optional[MCPManager] = None
        self._current_cycle_detail: Optional[CycleDetail] = None
        self._source_messages_by_id: dict[str, SessionMessage] = {}
        self._running = False
        self._cycle_counter = 0
        self._internal_loop_task: Optional[asyncio.Task] = None
        self._loop_task: Optional[asyncio.Task] = None
        self._new_message_event = asyncio.Event()
        self._max_internal_rounds = global_config.maisaka.max_internal_rounds
        self._max_context_size = max(1, int(global_config.chat.max_context_size))
        self._agent_state: Literal["running", "wait", "stop"] = self._STATE_STOP
        self._wait_until: Optional[float] = None
        self._reasoning_engine = MaisakaReasoningEngine(self)

    async def start(self) -> None:
        """Start the runtime loop."""
        if self._running:
            return

        if global_config.maisaka.enable_mcp:
            await self._init_mcp()

        self._running = True
        self._internal_loop_task = asyncio.create_task(self._reasoning_engine.run_loop())
        self._loop_task = asyncio.create_task(self._main_loop())
        logger.info(f"{self.log_prefix} MaiSaka 启动")

    async def stop(self) -> None:
        """Stop the runtime loop."""
        if not self._running:
            return

        self._running = False
        self._new_message_event.set()
        self.message_cache.clear()
        while not self._message_queue.empty():
            _ = self._message_queue.get_nowait()
        while not self._internal_turn_queue.empty():
            _ = self._internal_turn_queue.get_nowait()

        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            finally:
                self._loop_task = None

        if self._internal_loop_task is not None:
            self._internal_loop_task.cancel()
            try:
                await self._internal_loop_task
            except asyncio.CancelledError:
                pass
            finally:
                self._internal_loop_task = None

        if self._mcp_manager is not None:
            await self._mcp_manager.close()
            self._mcp_manager = None

        logger.info(f"{self.log_prefix} MaiSaka runtime stopped")

    def adjust_talk_frequency(self, frequency: float) -> None:
        """Compatibility shim for the existing manager API."""
        _ = frequency

    async def register_message(self, message: SessionMessage) -> None:
        """Append a newly received message into the HFC-style message cache."""
        self.message_cache.append(message)
        await self._message_queue.put(message)
        self._source_messages_by_id[message.message_id] = message
        if self._agent_state in (self._STATE_WAIT, self._STATE_STOP):
            self._agent_state = self._STATE_RUNNING
        self._new_message_event.set()

    async def _main_loop(self) -> None:
        try:
            while self._running:
                if self._message_queue.empty():
                    if self._agent_state == self._STATE_WAIT:
                        message_arrived = await self._wait_for_trigger()
                    else:
                        self._new_message_event.clear()
                        await self._new_message_event.wait()
                        message_arrived = self._running
                else:
                    message_arrived = True
                if not self._running:
                    return
                if not message_arrived:
                    self._agent_state = self._STATE_STOP
                    continue

                self._new_message_event.clear()

                # 加锁灌注消息
                while not self._message_queue.empty():
                    cached_messages = self._drain_message_cache()
                    if cached_messages:
                        await self._internal_turn_queue.put(cached_messages)
        except asyncio.CancelledError:
            logger.info(f"{self.log_prefix} MaiSaka runtime loop cancelled")

    async def _wait_for_trigger(self) -> bool:
        """等待外部触发。返回 True 表示有新消息事件，返回 False 表示等待超时。"""
        if self._agent_state != self._STATE_WAIT:
            await self._new_message_event.wait()
            return True

        # 处理 wait 工具调用带来的等待窗口：超时后恢复 idle；有新消息则继续处理缓存消息
        if self._wait_until is None:
            await self._new_message_event.wait()
            return True

        timeout = self._wait_until - time.time()
        if timeout <= 0:
            logger.info(f"{self.log_prefix} Maisaka 等待超时，继续查看新消息")
            self._enter_stop_state()
            self._wait_until = None
            return False

        try:
            await asyncio.wait_for(self._new_message_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.info(f"{self.log_prefix} Maisaka 等待超时，继续查看新消息")
            self._enter_stop_state()
            self._wait_until = None
            return False

    def _enter_wait_state(self, seconds: Optional[float] = None) -> None:
        """进入等待状态，seconds 为 None 时表示一直等待直到新消息到达。"""
        self._agent_state = self._STATE_WAIT
        self._wait_until = None if seconds is None else time.time() + seconds

    def _enter_stop_state(self) -> None:
        """进入停顿状态：仅等待新消息。"""
        self._agent_state = self._STATE_STOP
        self._wait_until = None

    def _drain_message_cache(self) -> list[SessionMessage]:
        """Drain the current message cache as one processing batch."""
        drained_messages = list(self.message_cache)
        self.message_cache.clear()
        while not self._message_queue.empty():
            try:
                drained_messages.append(self._message_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return drained_messages

    async def _init_mcp(self) -> None:
        """Initialize MCP tools for the runtime and inject them into the planner."""
        config_path = Path(__file__).with_name("mcp_config.json")
        self._mcp_manager = await MCPManager.from_config(str(config_path))
        if self._mcp_manager is None:
            logger.info(f"{self.log_prefix} MCP manager is unavailable")
            return

        mcp_tools = self._mcp_manager.get_openai_tools()
        if not mcp_tools:
            logger.info(f"{self.log_prefix} No MCP tools were exposed to Maisaka")
            return

        self._llm_service.set_extra_tools(mcp_tools)
        logger.info(
            f"{self.log_prefix} Loaded {len(mcp_tools)} MCP tools into Maisaka:\n"
            f"{self._mcp_manager.get_tool_summary()}"
        )

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
                session_id=self.session_id,
                group_info=self._build_group_info(message),
                user_info=self._build_runtime_user_info(),
                raw_message=user_sequence,
                display_text=visible_text,
            )
            self._chat_history.append(history_message)
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
        """Start a Maisaka thinking cycle."""
        self._cycle_counter += 1
        self._current_cycle_detail = CycleDetail(cycle_id=self._cycle_counter)
        self._current_cycle_detail.thinking_id = f"maisaka_tid{round(time.time(), 2)}"
        return self._current_cycle_detail

    def _end_cycle(self, cycle_detail: CycleDetail, only_long_execution: bool = True) -> CycleDetail:
        """End and record a Maisaka thinking cycle."""
        cycle_detail.end_time = time.time()
        self.history_loop.append(cycle_detail)

        timer_strings = [
            f"{name}: {duration:.2f}s"
            for name, duration in cycle_detail.time_records.items()
            if not only_long_execution or duration >= 0.1
        ]
        logger.info(
            f"{self.log_prefix} MaiSaka cycle={cycle_detail.cycle_id} completed "
            f"in {cycle_detail.end_time - cycle_detail.start_time:.2f}s; "
            f"stages={', '.join(timer_strings) if timer_strings else 'none'}"
        )
        return cycle_detail

    def _trim_chat_history(self) -> None:
        """Trim the oldest history until the user-message count is below the configured limit."""
        user_message_count = sum(1 for message in self._chat_history if get_message_role(message) == "user")
        if user_message_count <= self._max_context_size:
            return

        trimmed_history = list(self._chat_history)
        removed_count = 0

        while user_message_count >= self._max_context_size and trimmed_history:
            removed_message = trimmed_history.pop(0)
            removed_count += 1
            if get_message_role(removed_message) == "user":
                user_message_count -= 1

        self._chat_history = trimmed_history
        logger.info(
            f"{self.log_prefix} Trimmed {removed_count} history messages; "
            f"remaining_user_messages={user_message_count}"
        )

    def _build_runtime_user_info(self) -> UserInfo:
        if self.chat_stream.user_id:
            return UserInfo(
                user_id=self.chat_stream.user_id,
                user_nickname=global_config.maisaka.user_name.strip() or "User",
                user_cardname=None,
            )
        return UserInfo(user_id="maisaka_user", user_nickname="user", user_cardname=None)

    def _build_runtime_bot_user_info(self) -> UserInfo:
        return UserInfo(
            user_id=str(global_config.bot.qq_account) if global_config.bot.qq_account else "maisaka_assistant",
            user_nickname=global_config.bot.nickname.strip() or "MaiSaka",
            user_cardname=None,
        )

    def _build_group_info(self, message: Optional[SessionMessage] = None) -> Optional[GroupInfo]:
        group_info = None
        if message is not None:
            group_info = message.message_info.group_info
        elif self.chat_stream.context and self.chat_stream.context.message:
            group_info = self.chat_stream.context.message.message_info.group_info

        if group_info is None:
            return None

        return GroupInfo(group_id=group_info.group_id, group_name=group_info.group_name)

    def _log_cycle_started(self, cycle_detail: CycleDetail, round_index: int) -> None:
        logger.info(
            f"{self.log_prefix} MaiSaka cycle={cycle_detail.cycle_id} "
            f"round={round_index + 1}/{self._max_internal_rounds} "
            f"context_size={len(self._chat_history)}"
        )

    def _log_cycle_completed(self, cycle_detail: CycleDetail, timer_strings: list[str]) -> None:
        logger.info(
            f"{self.log_prefix} MaiSaka cycle={cycle_detail.cycle_id} completed "
            f"in {cycle_detail.end_time - cycle_detail.start_time:.2f}s; "
            f"stages={', '.join(timer_strings) if timer_strings else 'none'}"
        )

    def _log_history_trimmed(self, removed_count: int, user_message_count: int) -> None:
        logger.info(
            f"{self.log_prefix} Trimmed {removed_count} history messages; "
            f"remaining_user_messages={user_message_count}"
        )

    def _log_internal_loop_cancelled(self) -> None:
        logger.info(f"{self.log_prefix} Maisaka internal loop cancelled")
