"""Maisaka runtime for non-CLI integrations."""

from pathlib import Path
from typing import Literal, Optional, cast

import asyncio
import time

from src.chat.heart_flow.heartFC_utils import CycleDetail
from src.chat.message_receive.chat_manager import BotChatSession, chat_manager
from src.chat.message_receive.message import SessionMessage
from src.common.data_models.mai_message_data_model import GroupInfo, UserInfo
from src.common.logger import get_logger
from src.common.utils.utils_config import ExpressionConfigUtils
from src.config.config import global_config
from src.know_u.knowledge import KnowledgeLearner
from src.learners.expression_learner import ExpressionLearner
from src.learners.jargon_miner import JargonMiner
from src.llm_models.payload_content.tool_option import ToolDefinitionInput
from src.mcp_module import MCPManager

from .chat_loop_service import MaisakaChatLoopService
from .context_messages import LLMContextMessage
from .reasoning_engine import MaisakaReasoningEngine

logger = get_logger("maisaka_runtime")


class MaisakaHeartFlowChatting:
    """Session-scoped Maisaka runtime."""

    _STATE_RUNNING: Literal["running"] = "running"
    _STATE_WAIT: Literal["wait"] = "wait"
    _STATE_STOP: Literal["stop"] = "stop"

    def __init__(self, session_id: str):
        self.session_id = session_id
        chat_stream = chat_manager.get_session_by_session_id(session_id)
        if chat_stream is None:
            raise ValueError(f"未找到会话 {session_id} 对应的 Maisaka 运行时")
        self.chat_stream: BotChatSession = chat_stream

        session_name = chat_manager.get_session_name(session_id) or session_id
        self.log_prefix = f"[{session_name}]"
        self._chat_loop_service = MaisakaChatLoopService()
        self._chat_history: list[LLMContextMessage] = []
        self.history_loop: list[CycleDetail] = []

        # Keep all original messages for batching and later learning.
        self.message_cache: list[SessionMessage] = []
        self._last_processed_index = 0
        self._internal_turn_queue: asyncio.Queue[Optional[list[SessionMessage]]] = asyncio.Queue()

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
        self._pending_wait_tool_call_id: Optional[str] = None
        self._planner_interrupt_flag: Optional[asyncio.Event] = None

        expr_use, jargon_learn, expr_learn = ExpressionConfigUtils.get_expression_config_for_chat(session_id)
        self._enable_expression_use = expr_use
        self._enable_expression_learning = expr_learn
        self._enable_jargon_learning = jargon_learn
        self._min_messages_for_extraction = 10
        self._min_extraction_interval = 30
        self._last_expression_extraction_time = 0.0
        self._last_knowledge_extraction_time = 0.0
        self._expression_learner = ExpressionLearner(session_id)
        self._jargon_miner = JargonMiner(session_id, session_name=session_name)
        self._knowledge_learner = KnowledgeLearner(session_id)

        self._reasoning_engine = MaisakaReasoningEngine(self)

    async def start(self) -> None:
        """Start the runtime loop."""
        if self._running:
            self._ensure_background_tasks_running()
            return

        if global_config.maisaka.enable_mcp:
            await self._init_mcp()

        self._running = True
        self._ensure_background_tasks_running()
        logger.info(f"{self.log_prefix} Maisaka 运行时已启动")

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

        logger.info(f"{self.log_prefix} Maisaka 运行时已停止")

    def adjust_talk_frequency(self, frequency: float) -> None:
        """Compatibility shim for the existing manager API."""
        _ = frequency

    async def register_message(self, message: SessionMessage) -> None:
        """Cache a new message and wake the main loop."""
        if self._running:
            self._ensure_background_tasks_running()
        self.message_cache.append(message)
        self._source_messages_by_id[message.message_id] = message
        if self._agent_state == self._STATE_RUNNING and self._planner_interrupt_flag is not None:
            logger.info(
                f"{self.log_prefix} 收到新消息，发起规划器打断; "
                f"消息编号={message.message_id} 缓存条数={len(self.message_cache)} "
                f"时间戳={time.time():.3f}"
            )
            self._planner_interrupt_flag.set()
        if self._agent_state in (self._STATE_WAIT, self._STATE_STOP):
            self._agent_state = self._STATE_RUNNING
        self._new_message_event.set()

    def _ensure_background_tasks_running(self) -> None:
        """确保后台任务仍在运行，若崩溃则自动拉起。"""
        if not self._running:
            return

        if self._internal_loop_task is None or self._internal_loop_task.done():
            if self._internal_loop_task is not None and not self._internal_loop_task.cancelled():
                try:
                    exc = self._internal_loop_task.exception()
                except Exception:
                    exc = None
                if exc is not None:
                    logger.error(f"{self.log_prefix} 内部循环任务异常退出: {exc}")
            self._internal_loop_task = asyncio.create_task(self._reasoning_engine.run_loop())
            logger.warning(f"{self.log_prefix} 已重新拉起 Maisaka 内部循环任务")

        if self._loop_task is None or self._loop_task.done():
            if self._loop_task is not None and not self._loop_task.cancelled():
                try:
                    exc = self._loop_task.exception()
                except Exception:
                    exc = None
                if exc is not None:
                    logger.error(f"{self.log_prefix} 主循环任务异常退出: {exc}")
            self._loop_task = asyncio.create_task(self._main_loop())
            logger.warning(f"{self.log_prefix} 已重新拉起 Maisaka 主循环任务")

    async def _main_loop(self) -> None:
        try:
            while self._running:
                if not self._has_pending_messages():
                    if self._agent_state == self._STATE_WAIT:
                        trigger_reason = await self._wait_for_trigger()
                    else:
                        self._new_message_event.clear()
                        await self._new_message_event.wait()
                        trigger_reason: Literal["message", "timeout", "stop"] = "message" if self._running else "stop"
                else:
                    trigger_reason = "message"

                if not self._running:
                    return
                if trigger_reason == "stop":
                    self._agent_state = self._STATE_STOP
                    continue

                self._new_message_event.clear()

                if trigger_reason == "timeout":
                    # 等待超时后继续下一轮内部思考，但不要重复注入旧消息。
                    logger.info(f"{self.log_prefix} 等待超时后已投递继续思考触发信号")
                    await self._internal_turn_queue.put(None)
                    continue

                while self._has_pending_messages():
                    cached_messages = self._collect_pending_messages()
                    if not cached_messages:
                        break
                    await self._internal_turn_queue.put(cached_messages)
                    asyncio.create_task(self._trigger_batch_learning(cached_messages))
        except asyncio.CancelledError:
            logger.info(f"{self.log_prefix} Maisaka 运行时主循环已取消")

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
            f"{self.log_prefix} 已从消息缓存区[{start_index}:{self._last_processed_index}] "
            f"收集 {len(unique_messages)} 条新消息"
        )
        return unique_messages

    async def _wait_for_trigger(self) -> Literal["message", "timeout", "stop"]:
        """等待 wait 状态的触发结果。"""
        if self._agent_state != self._STATE_WAIT:
            await self._new_message_event.wait()
            return "message"

        if self._wait_until is None:
            await self._new_message_event.wait()
            return "message"

        timeout = self._wait_until - time.time()
        if timeout <= 0:
            logger.info(f"{self.log_prefix} Maisaka 等待已超时")
            self._agent_state = self._STATE_RUNNING
            self._wait_until = None
            return "timeout"

        try:
            await asyncio.wait_for(self._new_message_event.wait(), timeout=timeout)
            return "message"
        except asyncio.TimeoutError:
            logger.info(f"{self.log_prefix} Maisaka 等待已超时")
            self._agent_state = self._STATE_RUNNING
            self._wait_until = None
            return "timeout"

    def _enter_wait_state(self, seconds: Optional[float] = None, tool_call_id: Optional[str] = None) -> None:
        """Enter wait state."""
        self._agent_state = self._STATE_WAIT
        self._wait_until = None if seconds is None else time.time() + seconds
        self._pending_wait_tool_call_id = tool_call_id

    def _enter_stop_state(self) -> None:
        """Enter stop state."""
        self._agent_state = self._STATE_STOP
        self._wait_until = None
        self._pending_wait_tool_call_id = None

    async def _trigger_batch_learning(self, messages: list[SessionMessage]) -> None:
        """按同一批消息触发表达方式、黑话和 knowledge 学习。"""
        expression_result, knowledge_result = await asyncio.gather(
            self._trigger_expression_learning(messages),
            self._trigger_knowledge_learning(messages),
            return_exceptions=True,
        )
        if isinstance(expression_result, Exception):
            logger.error(f"{self.log_prefix} 表达学习任务异常退出: {expression_result}")
        if isinstance(knowledge_result, Exception):
            logger.error(f"{self.log_prefix} 知识学习任务异常退出: {knowledge_result}")

    async def _trigger_expression_learning(self, messages: list[SessionMessage]) -> None:
        """Trigger expression learning from the newly collected batch."""
        self._expression_learner.add_messages(messages)

        if not self._enable_expression_learning:
            logger.debug(f"{self.log_prefix} 表达学习未启用，跳过当前批次")
            return

        elapsed = time.time() - self._last_expression_extraction_time
        if elapsed < self._min_extraction_interval:
            logger.debug(
                f"{self.log_prefix} 表达学习尚未达到触发间隔: "
                f"已过={elapsed:.2f} 秒 阈值={self._min_extraction_interval} 秒"
            )
            return

        cache_size = self._expression_learner.get_cache_size()
        if cache_size < self._min_messages_for_extraction:
            logger.debug(
                f"{self.log_prefix} 表达学习因缓存数量不足而跳过: "
                f"学习器缓存={cache_size} 阈值={self._min_messages_for_extraction} "
                f"消息总缓存={len(self.message_cache)}"
            )
            return

        self._last_expression_extraction_time = time.time()
        logger.info(
            f"{self.log_prefix} 开始表达学习: "
            f"新批次消息数={len(messages)} 学习器缓存={cache_size} "
            f"消息总缓存={len(self.message_cache)} "
            f"启用黑话学习={self._enable_jargon_learning}"
        )

        try:
            jargon_miner = self._jargon_miner if self._enable_jargon_learning else None
            learnt_style = await self._expression_learner.learn(jargon_miner)
            if learnt_style:
                logger.info(f"{self.log_prefix} 表达学习已完成")
            else:
                logger.debug(f"{self.log_prefix} 表达学习已完成，但没有可用结果")
        except Exception:
            logger.exception(f"{self.log_prefix} 表达学习失败")

    async def _trigger_knowledge_learning(self, messages: list[SessionMessage]) -> None:
        """Trigger knowledge learning from the newly collected batch."""
        self._knowledge_learner.add_messages(messages)

        if not global_config.maisaka.enable_knowledge_module:
            logger.debug(f"{self.log_prefix} 知识学习未启用，跳过当前批次")
            return

        elapsed = time.time() - self._last_knowledge_extraction_time
        if elapsed < self._min_extraction_interval:
            logger.debug(
                f"{self.log_prefix} 知识学习尚未达到触发间隔: "
                f"已过={elapsed:.2f} 秒 阈值={self._min_extraction_interval} 秒"
            )
            return

        cache_size = self._knowledge_learner.get_cache_size()
        if cache_size < self._min_messages_for_extraction:
            logger.debug(
                f"{self.log_prefix} 知识学习因缓存数量不足而跳过: "
                f"学习器缓存={cache_size} 阈值={self._min_messages_for_extraction} "
                f"消息总缓存={len(self.message_cache)}"
            )
            return

        self._last_knowledge_extraction_time = time.time()
        logger.info(
            f"{self.log_prefix} 开始知识学习: "
            f"新批次消息数={len(messages)} 学习器缓存={cache_size} "
            f"消息总缓存={len(self.message_cache)}"
        )

        try:
            added_count = await self._knowledge_learner.learn()
            if added_count > 0:
                logger.info(f"{self.log_prefix} 知识学习已完成: 新增条目数={added_count}")
            else:
                logger.debug(f"{self.log_prefix} 知识学习已完成，但没有可用结果")
        except Exception:
            logger.exception(f"{self.log_prefix} 知识学习失败")

    async def _init_mcp(self) -> None:
        """Initialize MCP tools and inject them into the planner."""
        config_path = Path(__file__).resolve().parents[2] / "config" / "mcp_config.json"
        self._mcp_manager = await MCPManager.from_config(str(config_path))
        if self._mcp_manager is None:
            logger.info(f"{self.log_prefix} MCP 管理器不可用")
            return

        mcp_tools = self._mcp_manager.get_openai_tools()
        if not mcp_tools:
            logger.info(f"{self.log_prefix} 没有可供 Maisaka 使用的 MCP 工具")
            return

        mcp_tool_definitions = [cast(ToolDefinitionInput, tool) for tool in mcp_tools]
        self._chat_loop_service.set_extra_tools(mcp_tool_definitions)
        logger.info(
            f"{self.log_prefix} 已向 Maisaka 加载 {len(mcp_tools)} 个 MCP 工具:\n"
            f"{self._mcp_manager.get_tool_summary()}"
        )

    def _build_runtime_user_info(self) -> UserInfo:
        if self.chat_stream.user_id:
            return UserInfo(
                user_id=self.chat_stream.user_id,
                user_nickname=global_config.maisaka.user_name.strip() or "用户",
                user_cardname=None,
            )
        return UserInfo(user_id="maisaka_user", user_nickname="用户", user_cardname=None)

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
            f"{self.log_prefix} MaiSaka 轮次开始: 循环编号={cycle_detail.cycle_id} "
            f"回合={round_index + 1}/{self._max_internal_rounds} "
            f"上下文消息数={len(self._chat_history)}"
        )

    def _log_cycle_completed(self, cycle_detail: CycleDetail, timer_strings: list[str]) -> None:
        end_time = cycle_detail.end_time if cycle_detail.end_time is not None else cycle_detail.start_time
        logger.info(
            f"{self.log_prefix} MaiSaka 轮次结束: 循环编号={cycle_detail.cycle_id} "
            f"总耗时={end_time - cycle_detail.start_time:.2f} 秒; "
            f"阶段耗时={', '.join(timer_strings) if timer_strings else '无'}"
        )

    def _log_history_trimmed(self, removed_count: int, user_message_count: int) -> None:
        logger.info(
            f"{self.log_prefix} 已裁剪 {removed_count} 条历史消息; "
            f"剩余计入上下文的消息数={user_message_count}"
        )

    def _log_internal_loop_cancelled(self) -> None:
        logger.info(f"{self.log_prefix} Maisaka 内部循环已取消")
