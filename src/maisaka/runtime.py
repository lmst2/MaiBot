"""
Maisaka runtime for non-CLI integrations.
"""

from datetime import datetime
from pathlib import Path
import time
from typing import Optional

import asyncio

from src.chat.heart_flow.heartFC_utils import CycleDetail
from src.chat.message_receive.chat_manager import BotChatSession, chat_manager
from src.chat.message_receive.message import SessionMessage
from src.common.data_models.mai_message_data_model import GroupInfo, MaiMessage, UserInfo
from src.common.data_models.message_component_data_model import MessageSequence
from src.common.logger import get_logger
from src.config.config import global_config
from src.llm_models.payload_content.tool_option import ToolCall
from src.services import send_service

from .config import (
    DIRECT_IMAGE_INPUT,
    ENABLE_KNOWLEDGE_MODULE,
    ENABLE_LIST_FILES,
    ENABLE_MCP,
    ENABLE_READ_FILE,
    ENABLE_WRITE_FILE,
    MERGE_USER_MESSAGES,
)
from .knowledge import retrieve_relevant_knowledge
from .llm_service import MaiSakaLLMService
from .mcp_client import MCPManager
from .message_adapter import (
    build_message,
    build_visible_text_from_sequence,
    clone_message_sequence,
    format_speaker_content,
    get_message_role,
    remove_last_perception,
)
from .tool_handlers import handle_list_files, handle_mcp_tool, handle_read_file, handle_unknown_tool, handle_write_file

logger = get_logger("maisaka_runtime")


class MaisakaHeartFlowChatting:
    """Session-scoped Maisaka runtime that replaces the HFC planner and reply loop."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.chat_stream: Optional[BotChatSession] = chat_manager.get_session_by_session_id(session_id)
        if self.chat_stream is None:
            raise ValueError(f"Session not found for Maisaka runtime: {session_id}")

        session_name = chat_manager.get_session_name(session_id) or session_id
        self.log_prefix = f"[{session_name}]"
        self._llm_service = MaiSakaLLMService(api_key="", base_url=None, model="")
        self._chat_history: list[MaiMessage] = []
        self.history_loop: list[CycleDetail] = []
        self.message_cache: list[SessionMessage] = []
        self._mcp_manager: Optional[MCPManager] = None
        self._current_cycle_detail: Optional[CycleDetail] = None
        self._source_messages_by_id: dict[str, SessionMessage] = {}
        self._running = False
        self._cycle_counter = 0
        self._loop_task: Optional[asyncio.Task] = None
        self._loop_lock = asyncio.Lock()
        self._new_message_event = asyncio.Event()
        self._max_internal_rounds = 6
        self._chat_start_time: Optional[datetime] = None
        self._last_user_input_time: Optional[datetime] = None
        self._last_assistant_response_time: Optional[datetime] = None
        self._user_input_times: list[datetime] = []
        self._max_context_size = max(1, int(global_config.chat.max_context_size))

    async def start(self) -> None:
        """Start the runtime loop."""
        if self._running:
            return

        if ENABLE_MCP:
            await self._init_mcp()

        self._running = True
        self._loop_task = asyncio.create_task(self._main_loop())
        logger.info(f"{self.log_prefix} MaiSaka runtime started")

    async def stop(self) -> None:
        """Stop the runtime loop."""
        if not self._running:
            return

        self._running = False
        self._new_message_event.set()

        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            finally:
                self._loop_task = None

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
        self._source_messages_by_id[message.message_id] = message
        self._new_message_event.set()

    async def _main_loop(self) -> None:
        try:
            while self._running:
                await self._new_message_event.wait()
                self._new_message_event.clear()

                async with self._loop_lock:
                    cached_messages = self._drain_message_cache()
                    if not cached_messages:
                        continue

                    await self._ingest_messages(cached_messages)
                    await self._run_internal_loop(anchor_message=cached_messages[-1])
        except asyncio.CancelledError:
            logger.info(f"{self.log_prefix} MaiSaka runtime loop cancelled")

    def _drain_message_cache(self) -> list[SessionMessage]:
        """Drain the current message cache as one processing batch."""
        drained_messages = list(self.message_cache)
        self.message_cache.clear()
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
        if self._chat_start_time is None:
            self._chat_start_time = messages[0].timestamp

        self._last_user_input_time = messages[-1].timestamp
        self._user_input_times.extend(message.timestamp for message in messages)

        if MERGE_USER_MESSAGES:
            merged_sequence = await self._merge_messages(messages)
            merged_content = build_visible_text_from_sequence(merged_sequence).strip()
            if not merged_sequence.components:
                return

            self._chat_history.append(
                build_message(
                    role="user",
                    content=merged_content,
                    source="user",
                    timestamp=messages[-1].timestamp,
                    platform=messages[-1].platform,
                    session_id=self.session_id,
                    group_info=self._build_group_info(messages[-1]),
                    user_info=self._build_runtime_user_info(),
                    raw_message=merged_sequence,
                    display_text=merged_content,
                )
            )
            self._trim_chat_history()
            return

        for message in messages:
            history_message = await self._build_user_history_message(message)
            if history_message is None:
                continue
            self._chat_history.append(history_message)
            self._trim_chat_history()

    async def _merge_messages(self, messages: list[SessionMessage]) -> MessageSequence:
        merged_sequence = MessageSequence([])

        for message in messages:
            user_info = message.message_info.user_info
            speaker_name = user_info.user_cardname or user_info.user_nickname or user_info.user_id
            prefix = format_speaker_content(speaker_name, "", message.timestamp, message.message_id)
            merged_sequence.text(prefix)

            appended_component = False
            if DIRECT_IMAGE_INPUT:
                source_sequence = getattr(message, "maisaka_original_raw_message", message.raw_message)
            else:
                source_sequence = message.raw_message

            for component in clone_message_sequence(source_sequence).components:
                merged_sequence.components.append(component)
                appended_component = True

            if not appended_component:
                if not message.processed_plain_text:
                    await message.process()
                content = (message.processed_plain_text or "").strip()
                if content:
                    merged_sequence.text(content)

            merged_sequence.text("\n")

        return merged_sequence

    async def _build_user_history_message(self, message: SessionMessage) -> Optional[MaiMessage]:
        user_sequence = await self._build_message_sequence(message)
        visible_text = build_visible_text_from_sequence(user_sequence).strip()
        if not user_sequence.components:
            return None

        return build_message(
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

    async def _build_message_sequence(self, message: SessionMessage) -> MessageSequence:
        message_sequence = MessageSequence([])
        user_info = message.message_info.user_info
        speaker_name = user_info.user_cardname or user_info.user_nickname or user_info.user_id
        message_sequence.text(format_speaker_content(speaker_name, "", message.timestamp, message.message_id))

        appended_component = False
        if DIRECT_IMAGE_INPUT:
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

    async def _run_internal_loop(self, anchor_message: SessionMessage) -> None:
        """Run the Maisaka internal loop, treating each thinking round as one cycle."""
        last_had_tool_calls = True

        for round_index in range(self._max_internal_rounds):
            cycle_detail = self._start_cycle()
            logger.info(
                f"{self.log_prefix} MaiSaka cycle={cycle_detail.cycle_id} "
                f"round={round_index + 1}/{self._max_internal_rounds} "
                f"context_size={len(self._chat_history)}"
            )
            try:
                if last_had_tool_calls:
                    perception_started_at = time.time()
                    await self._append_perception_snapshot()
                    cycle_detail.time_records["perception"] = time.time() - perception_started_at

                planner_started_at = time.time()
                response = await self._llm_service.chat_loop_step(self._chat_history)
                cycle_detail.time_records["planner"] = time.time() - planner_started_at

                response.raw_message.platform = anchor_message.platform
                response.raw_message.session_id = self.session_id
                response.raw_message.message_info.group_info = self._build_group_info(anchor_message)
                self._chat_history.append(response.raw_message)
                self._last_assistant_response_time = datetime.now()

                if response.tool_calls:
                    tool_started_at = time.time()
                    should_pause = await self._handle_tool_calls(response.tool_calls, response.content or "", anchor_message)
                    cycle_detail.time_records["tool_calls"] = time.time() - tool_started_at
                    if should_pause:
                        return
                    last_had_tool_calls = True
                    continue

                if response.content:
                    last_had_tool_calls = False
                    continue

                return
            finally:
                self._end_cycle(cycle_detail)

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

    async def _append_perception_snapshot(self) -> None:
        tasks = []
        if ENABLE_KNOWLEDGE_MODULE:
            tasks.append(("knowledge", retrieve_relevant_knowledge(self._llm_service, self._chat_history)))

        if not tasks:
            return

        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

        perception_parts: list[str] = []
        for (task_name, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                logger.warning(f"{self.log_prefix} Maisaka {task_name} analysis failed: {result}")
                continue
            if result:
                perception_parts.append(f"{task_name.title()}\n{result}")

        remove_last_perception(self._chat_history)
        if not perception_parts:
            return

        self._chat_history.append(
            build_message(
                role="assistant",
                content="\n\n".join(perception_parts),
                message_kind="perception",
                source="assistant",
                platform=self.chat_stream.platform,
                session_id=self.session_id,
                group_info=self._build_group_info(),
                user_info=self._build_runtime_bot_user_info(),
            )
        )

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
                self._chat_history.append(
                    self._build_tool_message(
                        tool_call,
                        "No visible reply was sent for this round.",
                    )
                )
                continue

            if tool_call.func_name == "wait":
                seconds = (tool_call.args or {}).get("seconds", 30)
                self._chat_history.append(
                    self._build_tool_message(
                        tool_call,
                        f"Waiting for future input for up to {seconds} seconds.",
                    )
                )
                return True

            if tool_call.func_name == "stop":
                self._chat_history.append(
                    self._build_tool_message(
                        tool_call,
                        "Conversation loop paused until a new message arrives.",
                    )
                )
                return True

            if tool_call.func_name == "write_file" and ENABLE_WRITE_FILE:
                await handle_write_file(tool_call, self._chat_history)
                continue

            if tool_call.func_name == "read_file" and ENABLE_READ_FILE:
                await handle_read_file(tool_call, self._chat_history)
                continue

            if tool_call.func_name == "list_files" and ENABLE_LIST_FILES:
                await handle_list_files(tool_call, self._chat_history)
                continue

            if self._mcp_manager and self._mcp_manager.is_mcp_tool(tool_call.func_name):
                await handle_mcp_tool(tool_call, self._chat_history, self._mcp_manager)
                continue

            await handle_unknown_tool(tool_call, self._chat_history)

        return False

    async def _handle_reply(self, tool_call: ToolCall, latest_thought: str, anchor_message: SessionMessage) -> bool:
        target_message_id = str((tool_call.args or {}).get("message_id", "")).strip()
        if not target_message_id:
            self._chat_history.append(
                self._build_tool_message(tool_call, "reply requires a valid message_id argument.")
            )
            return False

        target_message = self._source_messages_by_id.get(target_message_id)
        if target_message is None:
            self._chat_history.append(
                self._build_tool_message(tool_call, f"reply target message_id not found: {target_message_id}")
            )
            return False

        reply_text = await self._llm_service.generate_reply(latest_thought, self._chat_history)
        sent = await send_service.text_to_stream(
            text=reply_text,
            stream_id=self.session_id,
            set_reply=True,
            reply_message=target_message,
            typing=False,
        )
        tool_result = "Visible reply generated and sent." if sent else "Visible reply generation succeeded but send failed."
        self._chat_history.append(self._build_tool_message(tool_call, tool_result))
        if not sent:
            return False

        bot_name = global_config.bot.nickname.strip() or "MaiSaka"
        self._chat_history.append(
            build_message(
                role="user",
                content=format_speaker_content(bot_name, reply_text, datetime.now()),
                source="guided_reply",
                platform=target_message.platform or anchor_message.platform,
                session_id=self.session_id,
                group_info=self._build_group_info(target_message),
                user_info=self._build_runtime_user_info(),
            )
        )
        return True

    def _build_tool_message(self, tool_call: ToolCall, content: str) -> MaiMessage:
        return build_message(
            role="tool",
            content=content,
            source="tool",
            tool_call_id=tool_call.call_id,
            platform=self.chat_stream.platform,
            session_id=self.session_id,
            group_info=self._build_group_info(),
            user_info=UserInfo(user_id="maisaka_tool", user_nickname="tool", user_cardname=None),
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
