"""Maisaka runtime for non-CLI integrations."""

from pathlib import Path
from typing import Literal, Optional

import asyncio
import time

from src.chat.heart_flow.heartFC_utils import CycleDetail
from src.chat.message_receive.chat_manager import BotChatSession, chat_manager
from src.chat.message_receive.message import SessionMessage
from src.common.data_models.mai_message_data_model import GroupInfo, UserInfo
from src.common.logger import get_logger
from src.common.utils.utils_config import ExpressionConfigUtils
from src.config.config import global_config
from src.mcp_module import MCPManager
from src.learners.expression_learner import ExpressionLearner
from src.learners.jargon_miner import JargonMiner

from .chat_loop_service import MaisakaChatLoopService
from .reasoning_engine import MaisakaReasoningEngine

logger = get_logger("maisaka_runtime")


class MaisakaHeartFlowChatting:
    """Session-scoped Maisaka runtime."""

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
        self._chat_loop_service = MaisakaChatLoopService()
        self._chat_history: list[SessionMessage] = []
        self.history_loop: list[CycleDetail] = []

        # Keep all original messages for batching and later learning.
        self.message_cache: list[SessionMessage] = []
        self._last_processed_index = 0
        self._internal_turn_queue: asyncio.Queue[list[SessionMessage]] = asyncio.Queue()

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

        expr_use, jargon_learn, expr_learn = ExpressionConfigUtils.get_expression_config_for_chat(session_id)
        self._enable_expression_use = expr_use
        self._enable_expression_learning = expr_learn
        self._enable_jargon_learning = jargon_learn
        self._min_messages_for_extraction = 10
        self._min_extraction_interval = 30
        self._last_extraction_time = 0.0
        self._expression_learner = ExpressionLearner(session_id)
        self._jargon_miner = JargonMiner(session_id, session_name=session_name)

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
        logger.info(f"{self.log_prefix} Maisaka runtime started")

    async def stop(self) -> None:
        """Stop the runtime loop."""
        if not self._running:
            return

        self._running = False
        self._new_message_event.set()
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

        logger.info(f"{self.log_prefix} Maisaka runtime stopped")

    def adjust_talk_frequency(self, frequency: float) -> None:
        """Compatibility shim for the existing manager API."""
        _ = frequency

    async def register_message(self, message: SessionMessage) -> None:
        """Cache a new message and wake the main loop."""
        self.message_cache.append(message)
        self._source_messages_by_id[message.message_id] = message
        if self._agent_state in (self._STATE_WAIT, self._STATE_STOP):
            self._agent_state = self._STATE_RUNNING
        self._new_message_event.set()

    async def _main_loop(self) -> None:
        try:
            while self._running:
                if not self._has_pending_messages():
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

                while self._has_pending_messages():
                    cached_messages = self._collect_pending_messages()
                    if not cached_messages:
                        break
                    await self._internal_turn_queue.put(cached_messages)
                    asyncio.create_task(self._trigger_expression_learning(cached_messages))
        except asyncio.CancelledError:
            logger.info(f"{self.log_prefix} Maisaka runtime loop cancelled")

    def _has_pending_messages(self) -> bool:
        return self._last_processed_index < len(self.message_cache)

    def _collect_pending_messages(self) -> list[SessionMessage]:
        """Collect one batch of unprocessed messages from message_cache."""
        start_index = self._last_processed_index
        pending_messages = self.message_cache[start_index:]
        if not pending_messages:
            return []

        unique_messages: list[SessionMessage] = []
        seen_message_ids: set[str] = set()
        for message in pending_messages:
            message_id = message.message_id
            if message_id in seen_message_ids:
                continue
            seen_message_ids.add(message_id)
            unique_messages.append(message)

        self._last_processed_index = len(self.message_cache)
        logger.info(
            f"{self.log_prefix} collected {len(unique_messages)} new messages "
            f"from message_cache[{start_index}:{self._last_processed_index}]"
        )
        return unique_messages

    async def _wait_for_trigger(self) -> bool:
        """Return True on new message, False on timeout."""
        if self._agent_state != self._STATE_WAIT:
            await self._new_message_event.wait()
            return True

        if self._wait_until is None:
            await self._new_message_event.wait()
            return True

        timeout = self._wait_until - time.time()
        if timeout <= 0:
            logger.info(f"{self.log_prefix} Maisaka wait timed out")
            self._enter_stop_state()
            self._wait_until = None
            return False

        try:
            await asyncio.wait_for(self._new_message_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.info(f"{self.log_prefix} Maisaka wait timed out")
            self._enter_stop_state()
            self._wait_until = None
            return False

    def _enter_wait_state(self, seconds: Optional[float] = None) -> None:
        """Enter wait state."""
        self._agent_state = self._STATE_WAIT
        self._wait_until = None if seconds is None else time.time() + seconds

    def _enter_stop_state(self) -> None:
        """Enter stop state."""
        self._agent_state = self._STATE_STOP
        self._wait_until = None

    async def _trigger_expression_learning(self, messages: list[SessionMessage]) -> None:
        """Trigger expression learning from the newly collected batch."""
        self._expression_learner.add_messages(messages)

        if not self._enable_expression_learning:
            logger.debug(f"{self.log_prefix} expression learning disabled, skip this batch")
            return

        elapsed = time.time() - self._last_extraction_time
        if elapsed < self._min_extraction_interval:
            logger.debug(
                f"{self.log_prefix} expression learning interval not reached: "
                f"elapsed={elapsed:.2f}s threshold={self._min_extraction_interval}s"
            )
            return

        cache_size = self._expression_learner.get_cache_size()
        if cache_size < self._min_messages_for_extraction:
            logger.debug(
                f"{self.log_prefix} expression learning skipped due to cache size: "
                f"learner_cache={cache_size} threshold={self._min_messages_for_extraction} "
                f"message_cache_total={len(self.message_cache)}"
            )
            return

        self._last_extraction_time = time.time()
        logger.info(
            f"{self.log_prefix} starting expression learning: "
            f"new_batch={len(messages)} learner_cache={cache_size} "
            f"message_cache_total={len(self.message_cache)} "
            f"enable_jargon_learning={self._enable_jargon_learning}"
        )

        try:
            jargon_miner = self._jargon_miner if self._enable_jargon_learning else None
            learnt_style = await self._expression_learner.learn(jargon_miner)
            if learnt_style:
                logger.info(f"{self.log_prefix} expression learning finished")
            else:
                logger.debug(f"{self.log_prefix} expression learning finished without usable result")
        except Exception:
            logger.exception(f"{self.log_prefix} expression learning failed")

    async def _init_mcp(self) -> None:
        """Initialize MCP tools and inject them into the planner."""
        config_path = Path(__file__).resolve().parents[2] / "config" / "mcp_config.json"
        self._mcp_manager = await MCPManager.from_config(str(config_path))
        if self._mcp_manager is None:
            logger.info(f"{self.log_prefix} MCP manager is unavailable")
            return

        mcp_tools = self._mcp_manager.get_openai_tools()
        if not mcp_tools:
            logger.info(f"{self.log_prefix} No MCP tools were exposed to Maisaka")
            return

        self._chat_loop_service.set_extra_tools(mcp_tools)
        logger.info(
            f"{self.log_prefix} Loaded {len(mcp_tools)} MCP tools into Maisaka:\n"
            f"{self._mcp_manager.get_tool_summary()}"
        )

    def _build_runtime_user_info(self) -> UserInfo:
        if self.chat_stream.user_id:
            return UserInfo(
                user_id=self.chat_stream.user_id,
                user_nickname=global_config.maisaka.user_name.strip() or "User",
                user_cardname=None,
            )
        return UserInfo(user_id="maisaka_user", user_nickname="user", user_cardname=None)

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
