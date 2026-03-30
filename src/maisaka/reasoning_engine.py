"""Maisaka 推理引擎。"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

import asyncio
import difflib
import json
import time
import traceback

from sqlmodel import col, select

from src.chat.heart_flow.heartFC_utils import CycleDetail
from src.chat.message_receive.message import SessionMessage
from src.chat.replyer.replyer_manager import replyer_manager
from src.chat.utils.utils import process_llm_response
from src.common.data_models.message_component_data_model import MessageSequence, TextComponent
from src.common.database.database import get_db_session
from src.common.database.database_model import PersonInfo
from src.common.logger import get_logger
from src.config.config import global_config
from src.know_u.knowledge_store import get_knowledge_store
from src.learners.jargon_explainer import search_jargon
from src.llm_models.exceptions import ReqAbortException
from src.llm_models.payload_content.tool_option import ToolCall
from src.services import database_service as database_api, send_service

from .context_messages import (
    AssistantMessage,
    LLMContextMessage,
    SessionBackedMessage,
    ToolResultMessage,
)
from .message_adapter import (
    build_visible_text_from_sequence,
    clone_message_sequence,
    format_speaker_content,
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

    async def run_loop(self) -> None:
        """独立消费消息批次，并执行对应的内部思考轮次。"""
        try:
            while self._runtime._running:
                cached_messages = await self._runtime._internal_turn_queue.get()
                timeout_triggered = cached_messages is None
                if not timeout_triggered and not cached_messages:
                    self._runtime._internal_turn_queue.task_done()
                    continue

                self._runtime._agent_state = self._runtime._STATE_RUNNING
                if cached_messages:
                    self._append_wait_interrupted_message_if_needed()
                    await self._ingest_messages(cached_messages)
                    anchor_message = cached_messages[-1]
                else:
                    anchor_message = self._get_timeout_anchor_message()
                    if anchor_message is None:
                        logger.warning(
                            f"{self._runtime.log_prefix} 等待超时后缺少可复用的锚点消息，跳过本轮继续思考"
                        )
                        self._runtime._internal_turn_queue.task_done()
                        continue
                    logger.info(f"{self._runtime.log_prefix} 等待超时后开始新一轮思考")
                    self._runtime._chat_history.append(self._build_wait_timeout_message())
                    self._trim_chat_history()
                try:
                    for round_index in range(self._runtime._max_internal_rounds):
                        cycle_detail = self._start_cycle()
                        self._runtime._log_cycle_started(cycle_detail, round_index)
                        try:
                            planner_started_at = time.time()
                            logger.info(
                                f"{self._runtime.log_prefix} 规划器开始执行: "
                                f"回合={round_index + 1} "
                                f"历史消息数={len(self._runtime._chat_history)} "
                                f"开始时间={planner_started_at:.3f}"
                            )
                            interrupt_flag = asyncio.Event()
                            self._runtime._planner_interrupt_flag = interrupt_flag
                            self._runtime._chat_loop_service.set_interrupt_flag(interrupt_flag)
                            try:
                                response = await self._runtime._chat_loop_service.chat_loop_step(self._runtime._chat_history)
                            finally:
                                if self._runtime._planner_interrupt_flag is interrupt_flag:
                                    self._runtime._planner_interrupt_flag = None
                                self._runtime._chat_loop_service.set_interrupt_flag(None)
                            cycle_detail.time_records["planner"] = time.time() - planner_started_at
                            logger.info(
                                f"{self._runtime.log_prefix} 规划器执行完成: "
                                f"回合={round_index + 1} "
                                f"耗时={cycle_detail.time_records['planner']:.3f} 秒"
                            )

                            reasoning_content = response.content or ""
                            if self._should_replace_reasoning(reasoning_content):
                                response.content = "让我根据新情况重新思考："
                                response.raw_message.content = "让我根据新情况重新思考："
                                logger.info(f"{self._runtime.log_prefix} 当前思考与上一轮过于相似，已替换为重新思考提示")

                            self._last_reasoning_content = reasoning_content
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
                        except ReqAbortException:
                            interrupted_at = time.time()
                            logger.info(
                                f"{self._runtime.log_prefix} 规划器打断成功: "
                                f"回合={round_index + 1} "
                                f"开始时间={planner_started_at:.3f} "
                                f"打断时间={interrupted_at:.3f} "
                                f"耗时={interrupted_at - planner_started_at:.3f} 秒"
                            )
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
            logger.exception(f"{self._runtime.log_prefix} Maisaka 内部循环发生异常")
            logger.error(traceback.format_exc())
            raise

    def _get_timeout_anchor_message(self) -> Optional[SessionMessage]:
        """在 wait 超时后复用最近一条真实用户消息作为锚点。"""
        if self._runtime.message_cache:
            return self._runtime.message_cache[-1]
        return None

    def _build_wait_timeout_message(self) -> ToolResultMessage:
        """构造 wait 超时后的工具结果消息。"""
        tool_call_id = self._runtime._pending_wait_tool_call_id or "wait_timeout"
        self._runtime._pending_wait_tool_call_id = None
        return ToolResultMessage(
            content="等待已超时，期间没有收到新的用户输入。请基于现有上下文继续下一轮思考。",
            timestamp=datetime.now(),
            tool_call_id=tool_call_id,
            tool_name="wait",
        )

    def _append_wait_interrupted_message_if_needed(self) -> None:
        """如果 wait 被新消息打断，则补一条对应的工具结果消息。"""
        tool_call_id = self._runtime._pending_wait_tool_call_id
        if not tool_call_id:
            return

        self._runtime._pending_wait_tool_call_id = None
        self._runtime._chat_history.append(
            ToolResultMessage(
                content="等待过程被新的用户输入打断，已继续处理最新消息。",
                timestamp=datetime.now(),
                tool_call_id=tool_call_id,
                tool_name="wait",
            )
        )

    async def _ingest_messages(self, messages: list[SessionMessage]) -> None:
        """处理传入消息列表，将其转换为历史消息并加入聊天历史缓存。"""
        for message in messages:
            # 构建用户消息序列
            user_sequence, visible_text = await self._build_message_sequence(message)
            if not user_sequence.components:
                continue

            history_message = SessionBackedMessage.from_session_message(
                message,
                raw_message=user_sequence,
                visible_text=visible_text,
                source_kind="user",
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

    def _insert_chat_history_message(self, message: LLMContextMessage) -> int:
        """将消息按处理顺序追加到聊天历史末尾。"""
        self._runtime._chat_history.append(message)
        return len(self._runtime._chat_history) - 1

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
        conversation_message_count = sum(1 for message in self._runtime._chat_history if message.count_in_context)
        if conversation_message_count <= self._runtime._max_context_size:
            return

        trimmed_history = list(self._runtime._chat_history)
        removed_count = 0

        while conversation_message_count >= self._runtime._max_context_size and trimmed_history:
            removed_message = trimmed_history.pop(0)
            removed_count += 1
            if removed_message.count_in_context:
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
            logger.info(
                f"{self._runtime.log_prefix} 跳过思考相似度判定: "
                f"上一轮为空={not bool(self._last_reasoning_content)} "
                f"当前为空={not bool(current_content)} 相似度=0.00"
            )
            return False

        similarity = self._calculate_similarity(current_content, self._last_reasoning_content)
        logger.info(f"{self._runtime.log_prefix} 思考内容相似度: {similarity:.2f}")
        return similarity > 0.9

    @staticmethod
    def _post_process_reply_text(reply_text: str) -> list[str]:
        """沿用旧回复链的文本后处理，执行分段与错别字注入。"""
        processed_segments: list[str] = []
        for segment in process_llm_response(reply_text):
            normalized_segment = segment.strip()
            if normalized_segment:
                processed_segments.append(normalized_segment)

        if processed_segments:
            return processed_segments
        return [reply_text.strip()]

    async def _handle_tool_calls(
        self,
        tool_calls: list[ToolCall],
        latest_thought: str,
        anchor_message: SessionMessage,
    ) -> bool:
        for tool_call in tool_calls:
            if tool_call.func_name == "reply":
                reply_sent = await self._handle_reply(tool_call, latest_thought, anchor_message)
                if not reply_sent:
                    logger.warning(
                        f"{self._runtime.log_prefix} 回复工具未生成可见消息，将继续下一轮循环"
                    )
                continue

            if tool_call.func_name == "no_reply":
                self._runtime._chat_history.append(
                    self._build_tool_message(
                        tool_call,
                        "本轮未发送可见回复。",
                    )
                )
                continue

            if tool_call.func_name == "query_jargon":
                await self._handle_query_jargon(tool_call)
                continue

            if tool_call.func_name == "query_person_info":
                await self._handle_query_person_info(tool_call)
                continue

            if tool_call.func_name == "wait":
                seconds = (tool_call.args or {}).get("seconds", 30)
                try:
                    wait_seconds = int(seconds)
                except (TypeError, ValueError):
                    wait_seconds = 30
                wait_seconds = max(0, wait_seconds)
                self._runtime._enter_wait_state(seconds=wait_seconds, tool_call_id=tool_call.call_id)
                return True

            if tool_call.func_name == "stop":
                self._runtime._chat_history.append(
                    self._build_tool_message(
                        tool_call,
                        "当前对话循环已暂停，等待新消息到来。",
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
                self._build_tool_message(tool_call, "查询黑话工具需要提供 `words` 数组参数。")
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
                self._build_tool_message(tool_call, "查询黑话工具至少需要一个非空词条。")
            )
            return

        logger.info(f"{self._runtime.log_prefix} 已触发黑话查询: 词条={words!r}")

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

        logger.info(f"{self._runtime.log_prefix} 黑话查询完成: 结果={results!r}")
        self._runtime._chat_history.append(
            self._build_tool_message(
                tool_call,
                json.dumps({"results": results}, ensure_ascii=False),
            )
        )

    async def _handle_query_person_info(self, tool_call: ToolCall) -> None:
        """查询指定人物的档案和相关知识。"""
        tool_args = tool_call.args or {}
        raw_person_name = tool_args.get("person_name")
        raw_limit = tool_args.get("limit", 3)

        if not isinstance(raw_person_name, str):
            self._runtime._chat_history.append(
                self._build_tool_message(tool_call, "查询人物信息工具需要提供字符串类型的 `person_name` 参数。")
            )
            return

        person_name = raw_person_name.strip()
        if not person_name:
            self._runtime._chat_history.append(
                self._build_tool_message(tool_call, "查询人物信息工具需要提供非空的 `person_name` 参数。")
            )
            return

        try:
            limit = max(1, min(int(raw_limit), 10))
        except (TypeError, ValueError):
            limit = 3

        logger.info(
            f"{self._runtime.log_prefix} 已触发人物信息查询: "
            f"人物名={person_name!r} 限制条数={limit}"
        )

        persons = self._query_person_records(person_name, limit)
        result = {
            "query": person_name,
            "persons": persons,
            "related_knowledge": self._query_related_knowledge(person_name, persons, limit),
        }

        logger.info(
            f"{self._runtime.log_prefix} 人物信息查询完成: "
            f"人物记录数={len(result['persons'])} 相关知识数={len(result['related_knowledge'])}"
        )
        self._runtime._chat_history.append(
            self._build_tool_message(
                tool_call,
                json.dumps(result, ensure_ascii=False),
            )
        )

    def _query_person_records(self, person_name: str, limit: int) -> list[dict[str, Any]]:
        """按名称、昵称或用户 ID 查询人物档案。"""
        with get_db_session() as session:
            records = session.exec(
                select(PersonInfo)
                .where(
                    col(PersonInfo.person_name).contains(person_name)
                    | col(PersonInfo.user_nickname).contains(person_name)
                    | col(PersonInfo.user_id).contains(person_name)
                )
                .order_by(col(PersonInfo.last_known_time).desc(), col(PersonInfo.id).desc())
                .limit(limit)
            ).all()

        persons: list[dict[str, Any]] = []
        for record in records:
            memory_points: list[str] = []
            if record.memory_points:
                try:
                    parsed_points = json.loads(record.memory_points)
                    if isinstance(parsed_points, list):
                        memory_points = [str(point).strip() for point in parsed_points if str(point).strip()]
                except (json.JSONDecodeError, TypeError, ValueError):
                    memory_points = []

            persons.append(
                {
                    "person_id": record.person_id,
                    "person_name": record.person_name or "",
                    "user_nickname": record.user_nickname,
                    "user_id": record.user_id,
                    "platform": record.platform,
                    "name_reason": record.name_reason or "",
                    "is_known": record.is_known,
                    "know_counts": record.know_counts,
                    "memory_points": memory_points[:20],
                    "last_known_time": (
                        record.last_known_time.isoformat() if record.last_known_time is not None else None
                    ),
                }
            )

        return persons

    def _query_related_knowledge(
        self,
        person_name: str,
        persons: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        """从 Maisaka knowledge 中补充检索与该人物相关的条目。"""
        store = get_knowledge_store()
        knowledge_items: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for person in persons:
            matched_items = store.get_knowledge_by_user(
                platform=str(person.get("platform", "")).strip(),
                user_id=str(person.get("user_id", "")).strip(),
                user_nickname=str(person.get("user_nickname", "")).strip(),
                person_name=str(person.get("person_name", "")).strip(),
                limit=max(limit, 5),
            )
            for item in matched_items:
                item_id = str(item.get("id", "")).strip()
                if item_id and item_id in seen_ids:
                    continue
                if item_id:
                    seen_ids.add(item_id)
                knowledge_items.append(item)

        if not knowledge_items:
            fallback_items = store.search_knowledge(person_name, limit=max(limit, 5))
            for item in fallback_items:
                item_id = str(item.get("id", "")).strip()
                if item_id and item_id in seen_ids:
                    continue
                if item_id:
                    seen_ids.add(item_id)
                knowledge_items.append(item)

        results: list[dict[str, Any]] = []
        for item in knowledge_items:
            results.append(
                {
                    "id": str(item.get("id", "")).strip(),
                    "category_id": str(item.get("category_id", "")).strip(),
                    "category_name": str(item.get("category_name", "")).strip(),
                    "content": str(item.get("content", "")).strip(),
                    "metadata": item.get("metadata", {}),
                    "created_at": item.get("created_at"),
                }
            )
        return results

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
                self._build_tool_message(tool_call, "回复工具需要提供有效的 `msg_id` 参数。")
            )
            return False

        target_message = self._runtime._source_messages_by_id.get(target_message_id)
        if target_message is None:
            self._runtime._chat_history.append(
                self._build_tool_message(tool_call, f"未找到要回复的目标消息，msg_id={target_message_id}")
            )
            return False

        logger.info(
            f"{self._runtime.log_prefix} 已触发回复工具: "
            f"目标消息编号={target_message_id} 引用回复={quote_reply} 最新思考={latest_thought!r}"
        )
        logger.info(f"{self._runtime.log_prefix} 正在获取 Maisaka 回复生成器")
        try:
            replyer = replyer_manager.get_replyer(
                chat_stream=self._runtime.chat_stream,
                request_type="maisaka_replyer",
                replyer_type="maisaka",
            )
        except Exception:
            logger.exception(
                f"{self._runtime.log_prefix} 获取回复生成器时发生异常: "
                f"目标消息编号={target_message_id}"
            )
            self._runtime._chat_history.append(
                self._build_tool_message(tool_call, "获取 Maisaka 回复生成器时发生异常。")
            )
            return False

        if replyer is None:
            logger.error(f"{self._runtime.log_prefix} 获取 Maisaka 回复生成器失败")
            self._runtime._chat_history.append(
                self._build_tool_message(tool_call, "Maisaka 回复生成器当前不可用。")
            )
            return False

        logger.info(f"{self._runtime.log_prefix} 已成功获取 Maisaka 回复生成器")

        logger.info(f"{self._runtime.log_prefix} 正在调用回复生成接口: 目标消息编号={target_message_id}")
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
                f"{self._runtime.log_prefix} 回复生成器执行异常: 目标消息编号={target_message_id} "
                f"异常类型={type(exc).__name__} 异常信息={str(exc)}\n{traceback.format_exc()}"
            )
            self._runtime._chat_history.append(
                self._build_tool_message(tool_call, "生成可见回复时发生异常。")
            )
            return False

        logger.info(
            f"{self._runtime.log_prefix} 回复生成完成: "
            f"成功={success} 回复文本={reply_result.completion.response_text!r} "
            f"错误信息={reply_result.error_message!r}"
        )
        reply_text = reply_result.completion.response_text.strip() if success else ""
        if not reply_text:
            logger.warning(
                f"{self._runtime.log_prefix} 回复生成器返回空文本: "
                f"目标消息编号={target_message_id} 错误信息={reply_result.error_message!r}"
            )
            self._runtime._chat_history.append(
                self._build_tool_message(tool_call, "生成可见回复失败。")
            )
            return False

        reply_segments = self._post_process_reply_text(reply_text)
        combined_reply_text = "".join(reply_segments)
        logger.info(
            f"{self._runtime.log_prefix} 回复后处理完成: "
            f"目标消息编号={target_message_id} 分段数={len(reply_segments)} "
            f"分段内容={reply_segments!r}"
        )

        logger.info(
            f"{self._runtime.log_prefix} 正在发送引导回复: "
            f"目标消息编号={target_message_id} 引用回复={quote_reply} 回复分段={reply_segments!r}"
        )
        try:
            sent = False
            for index, segment in enumerate(reply_segments):
                sent = await send_service.text_to_stream(
                    text=segment,
                    stream_id=self._runtime.session_id,
                    set_reply=quote_reply if index == 0 else False,
                    reply_message=target_message if quote_reply and index == 0 else None,
                    selected_expressions=reply_result.selected_expression_ids or None,
                    typing=index > 0,
                )
                if not sent:
                    break
        except Exception:
            logger.exception(
                f"{self._runtime.log_prefix} 发送文字消息时发生异常，目标消息编号={target_message_id}"
            )
            self._runtime._chat_history.append(
                self._build_tool_message(tool_call, "发送可见回复时发生异常。")
            )
            return False

        logger.info(
            f"{self._runtime.log_prefix} 引导回复发送结果: "
            f"目标消息编号={target_message_id} 发送成功={sent}"
        )
        tool_result = "可见回复已生成并发送。" if sent else "可见回复生成成功，但发送失败。"
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
                display_prompt=f"你对{target_user_name}进行了回复：{combined_reply_text}",
                tool_data={
                    "msg_id": target_message_id,
                    "quote": quote_reply,
                    "reply_text": combined_reply_text,
                    "reply_segments": reply_segments,
                },
                tool_name="reply",
                tool_reasoning=latest_thought,
            )

        bot_name = global_config.bot.nickname.strip() or "MaiSaka"
        reply_timestamp = datetime.now()
        planner_prefix = (
            f"[时间]{reply_timestamp.strftime('%H:%M:%S')}\n"
            f"[用户]{bot_name}\n"
            "[用户群昵称]\n"
            "[msg_id]\n"
            "[发言内容]"
        )
        history_message = SessionBackedMessage(
            raw_message=MessageSequence([TextComponent(f"{planner_prefix}{combined_reply_text}")]),
            visible_text="",
            timestamp=reply_timestamp,
            source_kind="guided_reply",
        )
        visible_reply_text = format_speaker_content(
            bot_name,
            combined_reply_text,
            reply_timestamp,
        )
        history_message.visible_text = visible_reply_text
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

        logger.info(f"{self._runtime.log_prefix} 已触发表情包发送工具: 情绪={emotion!r}")

        # 获取表情包列表
        if not emoji_manager.emojis:
            self._runtime._chat_history.append(
                self._build_tool_message(tool_call, "当前表情包库中没有可用表情。")
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
                    f"{self._runtime.log_prefix} 找到 {len(matching_emojis)} 个匹配情绪 {emotion!r} 的表情包，"
                    f"已选择：{selected_emoji.description}"
                )

        # 如果没有找到匹配的情感表情包，随机选择一个
        if selected_emoji is None:
            selected_emoji = random.choice(emoji_manager.emojis)
            logger.info(
                f"{self._runtime.log_prefix} 没有表情包匹配情绪 {emotion!r}，"
                f"已随机选择：{selected_emoji.description}"
            )

        # 更新表情包使用次数
        emoji_manager.update_emoji_usage(selected_emoji)

        # 获取表情包的 base64 数据
        try:
            emoji_base64 = ImageUtils.image_path_to_base64(str(selected_emoji.full_path))
            if not emoji_base64:
                raise ValueError("表情图片转换为 base64 失败")
        except Exception as exc:
            logger.error(
                f"{self._runtime.log_prefix} 表情图片转换为 base64 失败: {exc}"
            )
            self._runtime._chat_history.append(
                self._build_tool_message(tool_call, f"发送表情包失败：{exc}")
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
                f"{self._runtime.log_prefix} 发送表情包时发生异常: {exc}"
            )
            self._runtime._chat_history.append(
                self._build_tool_message(tool_call, f"发送表情包时发生异常：{exc}")
            )
            return

        if sent:
            logger.info(
                f"{self._runtime.log_prefix} 表情包发送成功: "
                f"描述={selected_emoji.description!r} 情绪标签={selected_emoji.emotion}"
            )
            self._runtime._chat_history.append(
                self._build_tool_message(
                    tool_call,
                    f"已发送表情包：{selected_emoji.description}（情绪：{', '.join(selected_emoji.emotion)}）"
                )
            )
        else:
            logger.warning(f"{self._runtime.log_prefix} 表情包发送失败")
            self._runtime._chat_history.append(
                self._build_tool_message(tool_call, "发送表情包失败。")
            )

    def _build_tool_message(self, tool_call: ToolCall, content: str) -> ToolResultMessage:
        return ToolResultMessage(
            content=content,
            timestamp=datetime.now(),
            tool_call_id=tool_call.call_id,
            tool_name=tool_call.func_name,
        )
