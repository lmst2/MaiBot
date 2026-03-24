from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import asyncio
import random
import time
import traceback

from rich.traceback import install

from src.learners.expression_learner import ExpressionLearner
from src.learners.jargon_miner import JargonMiner
from src.chat.event_helpers import build_event_message
from src.chat.logger.plan_reply_logger import PlanReplyLogger
from src.chat.message_receive.chat_manager import BotChatSession
from src.chat.message_receive.chat_manager import chat_manager as _chat_manager
from src.chat.planner_actions.action_manager import ActionManager
from src.chat.planner_actions.action_modifier import ActionModifier
from src.chat.planner_actions.planner import ActionPlanner
from src.chat.utils.prompt_builder import global_prompt_manager
from src.chat.utils.timer_calculator import Timer
from src.chat.utils.utils import record_replyer_action_temp
from src.common.data_models.info_data_model import ActionPlannerInfo
from src.common.data_models.message_component_data_model import MessageSequence, TextComponent
from src.common.logger import get_logger
from src.common.utils.utils_config import ChatConfigUtils, ExpressionConfigUtils
from src.config.config import global_config
from src.config.file_watcher import FileChange
from src.core.event_bus import event_bus
from src.core.types import ActionInfo, EventType
from src.person_info.person_info import Person
from src.services import (
    database_service as database_api,
    generator_service as generator_api,
    message_service as message_api,
    send_service as send_api,
)
from src.services.message_service import build_readable_messages_with_id, get_messages_before_time_in_chat

from .heartFC_utils import CycleDetail

if TYPE_CHECKING:
    from src.chat.message_receive.message import SessionMessage


install(extra_lines=5)

logger = get_logger("heartFC_chat")


class HeartFChatting:
    """管理一个持续运行的 Focus Chat 会话。"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.chat_stream: BotChatSession = _chat_manager.get_session_by_session_id(self.session_id)  # type: ignore[assignment]
        if not self.chat_stream:
            raise ValueError(f"无法找到聊天会话 {self.session_id}")

        session_name = _chat_manager.get_session_name(session_id) or session_id
        self.log_prefix = f"[{session_name}]"
        self.session_name = session_name

        self.action_manager = ActionManager()
        self.action_planner = ActionPlanner(chat_id=self.session_id, action_manager=self.action_manager)
        self.action_modifier = ActionModifier(action_manager=self.action_manager, chat_id=self.session_id)

        self._running: bool = False
        self._loop_task: Optional[asyncio.Task] = None
        self._cycle_event = asyncio.Event()
        self._hfc_lock = asyncio.Lock()

        self._cycle_counter = 0
        self._current_cycle_detail: Optional[CycleDetail] = None
        self.history_loop: List[CycleDetail] = []

        self.last_read_time = time.time() - 2
        self.last_active_time = time.time()
        self._talk_frequency_adjust = 1.0
        self._consecutive_no_reply_count = 0

        self.message_cache: List["SessionMessage"] = []

        self._min_messages_for_extraction = 30
        self._min_extraction_interval = 60
        self._last_extraction_time = 0.0

        expr_use, jargon_learn, expr_learn = ExpressionConfigUtils.get_expression_config_for_chat(session_id)
        self._enable_expression_use = expr_use
        self._enable_expression_learning = expr_learn
        self._enable_jargon_learning = jargon_learn
        self._expression_learner = ExpressionLearner(session_id)
        self._jargon_miner = JargonMiner(session_id, session_name=session_name)

    async def start(self):
        if self._running:
            logger.debug(f"{self.log_prefix} HeartFChatting 已在运行中")
            return

        try:
            self._running = True
            self._cycle_event.clear()
            self._loop_task = asyncio.create_task(self.main_loop())
            self._loop_task.add_done_callback(self._handle_loop_completion)
            logger.info(f"{self.log_prefix} HeartFChatting 启动完成")
        except Exception as exc:
            logger.error(f"{self.log_prefix} HeartFChatting 启动失败: {exc}", exc_info=True)
            self._running = False
            self._cycle_event.set()
            self._loop_task = None
            raise

    async def stop(self):
        if not self._running:
            logger.debug(f"{self.log_prefix} HeartFChatting 已停止")
            return

        self._running = False
        self._cycle_event.set()

        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                logger.info(f"{self.log_prefix} HeartFChatting 主循环已取消")
            except Exception as exc:
                logger.error(f"{self.log_prefix} 停止 HeartFChatting 时发生错误: {exc}", exc_info=True)
            finally:
                self._loop_task = None

        logger.info(f"{self.log_prefix} HeartFChatting 已停止")

    def adjust_talk_frequency(self, new_value: float):
        self._talk_frequency_adjust = max(0.0, new_value)

    async def register_message(self, message: "SessionMessage"):
        self.message_cache.append(message)

        if global_config.chat.inevitable_at_reply and message.is_at:
            self.last_read_time = time.time()
            async with self._hfc_lock:
                await self._judge_and_response(mentioned_message=message, recent_messages_list=[message])
            return

        if global_config.chat.mentioned_bot_reply and message.is_mentioned:
            self.last_read_time = time.time()
            async with self._hfc_lock:
                await self._judge_and_response(mentioned_message=message, recent_messages_list=[message])
            return

    async def main_loop(self):
        try:
            while self._running and not self._cycle_event.is_set():
                if not self._hfc_lock.locked():
                    async with self._hfc_lock:
                        await self._hfc_func()
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            logger.info(f"{self.log_prefix} HeartFChatting: 主循环被取消")
        except Exception as exc:
            logger.error(f"{self.log_prefix} HeartFChatting: 主循环异常: {exc}", exc_info=True)
            await self.stop()
            await asyncio.sleep(3)
            await self.start()

    async def _config_callback(self, file_change: Optional[FileChange] = None):
        del file_change
        expr_use, jargon_learn, expr_learn = ExpressionConfigUtils.get_expression_config_for_chat(self.session_id)
        self._enable_expression_use = expr_use
        self._enable_expression_learning = expr_learn
        self._enable_jargon_learning = jargon_learn

    async def _hfc_func(self):
        recent_messages_list = message_api.get_messages_by_time_in_chat(
            chat_id=self.session_id,
            start_time=self.last_read_time,
            end_time=time.time(),
            limit=20,
            limit_mode="latest",
            filter_mai=True,
            filter_command=False,
            filter_intercept_message_level=1,
        )

        if len(recent_messages_list) < 1:
            await asyncio.sleep(0.2)
            return True

        self.last_read_time = time.time()

        mentioned_message: Optional["SessionMessage"] = None
        for message in recent_messages_list:
            if global_config.chat.inevitable_at_reply and message.is_at:
                mentioned_message = message
            elif global_config.chat.mentioned_bot_reply and message.is_mentioned:
                mentioned_message = message

        talk_value = ChatConfigUtils.get_talk_value(self.session_id) * self._talk_frequency_adjust
        if mentioned_message:
            await self._judge_and_response(mentioned_message=mentioned_message, recent_messages_list=recent_messages_list)
        elif random.random() < talk_value:
            await self._judge_and_response(recent_messages_list=recent_messages_list)
        return True

    async def _judge_and_response(
        self,
        mentioned_message: Optional["SessionMessage"] = None,
        recent_messages_list: Optional[List["SessionMessage"]] = None,
    ):
        recent_messages = list(recent_messages_list or self.message_cache[-20:])
        if recent_messages:
            asyncio.create_task(self._trigger_expression_learning(recent_messages))

        cycle_timers, thinking_id = self._start_cycle()
        logger.info(f"{self.log_prefix} 开始第{self._cycle_counter}次思考")

        try:
            async with global_prompt_manager.async_message_scope(self._get_template_name()):
                available_actions: Dict[str, ActionInfo] = {}
                try:
                    await self.action_modifier.modify_actions()
                    available_actions = self.action_manager.get_using_actions()
                except Exception as exc:
                    logger.error(f"{self.log_prefix} 动作修改失败: {exc}", exc_info=True)

                is_group_chat, chat_target_info, _ = self.action_planner.get_necessary_info()
                message_list_before_now = get_messages_before_time_in_chat(
                    chat_id=self.session_id,
                    timestamp=time.time(),
                    limit=int(global_config.chat.max_context_size * 0.6),
                    filter_intercept_message_level=1,
                )
                chat_content_block, message_id_list = build_readable_messages_with_id(
                    messages=message_list_before_now,
                    timestamp_mode="normal_no_YMD",
                    read_mark=self.action_planner.last_obs_time_mark,
                    truncate=True,
                    show_actions=True,
                )

                prompt, filtered_actions = await self._build_planner_prompt_with_event(
                    available_actions=available_actions,
                    is_group_chat=is_group_chat,
                    chat_target_info=chat_target_info,
                    chat_content_block=chat_content_block,
                    message_id_list=message_id_list,
                )
                if prompt is None:
                    return False

                with Timer("规划器", cycle_timers):
                    reasoning, action_to_use_info, llm_raw_output, llm_reasoning, llm_duration_ms = (
                        await self.action_planner._execute_main_planner(
                            prompt=prompt,
                            message_id_list=message_id_list,
                            filtered_actions=filtered_actions,
                            available_actions=available_actions,
                            loop_start_time=self.last_read_time,
                        )
                    )

                action_to_use_info = self._ensure_force_reply_action(
                    actions=action_to_use_info,
                    force_reply_message=mentioned_message,
                    available_actions=available_actions,
                )
                self.action_planner.add_plan_log(reasoning, action_to_use_info)
                self.action_planner.last_obs_time_mark = time.time()
                self._log_plan(
                    prompt=prompt,
                    reasoning=reasoning,
                    llm_raw_output=llm_raw_output,
                    llm_reasoning=llm_reasoning,
                    llm_duration_ms=llm_duration_ms,
                    actions=action_to_use_info,
                )

                logger.info(
                    f"{self.log_prefix} 决定执行{len(action_to_use_info)}个动作: {' '.join([a.action_type for a in action_to_use_info])}"
                )

                action_tasks = [
                    asyncio.create_task(
                        self._execute_action(
                            action,
                            action_to_use_info,
                            thinking_id,
                            available_actions,
                            cycle_timers,
                        )
                    )
                    for action in action_to_use_info
                ]
                results = await asyncio.gather(*action_tasks, return_exceptions=True)

                reply_loop_info = None
                reply_text_from_reply = ""
                action_success = False
                action_reply_text = ""
                execute_result_str = ""

                for result in results:
                    if isinstance(result, BaseException):
                        logger.error(f"{self.log_prefix} 动作执行异常: {result}", exc_info=True)
                        continue

                    execute_result_str += f"{result['action_type']} 执行结果:{result['result']}\n"
                    if result["action_type"] == "reply":
                        if result["success"]:
                            reply_loop_info = result["loop_info"]
                            reply_text_from_reply = result["result"]
                        else:
                            logger.warning(f"{self.log_prefix} reply 动作执行失败")
                    else:
                        action_success = result["success"]
                        action_reply_text = result["result"]

                self.action_planner.add_plan_excute_log(result=execute_result_str)

                if reply_loop_info:
                    loop_info = reply_loop_info
                    loop_info["loop_action_info"].update(
                        {
                            "action_taken": action_success,
                            "taken_time": time.time(),
                        }
                    )
                else:
                    loop_info = {
                        "loop_plan_info": {
                            "action_result": action_to_use_info,
                        },
                        "loop_action_info": {
                            "action_taken": action_success,
                            "reply_text": action_reply_text,
                            "taken_time": time.time(),
                        },
                    }
                    reply_text_from_reply = action_reply_text

                current_cycle_detail = self._end_cycle(self._current_cycle_detail, loop_info)
                logger.debug(f"{self.log_prefix} 本轮最终输出: {reply_text_from_reply}")
                return current_cycle_detail is not None
        except Exception as exc:
            logger.error(f"{self.log_prefix} 判定与回复流程失败: {exc}", exc_info=True)
            if self._current_cycle_detail:
                self._end_cycle(
                    self._current_cycle_detail,
                    {
                        "loop_plan_info": {"action_result": []},
                        "loop_action_info": {
                            "action_taken": False,
                            "reply_text": "",
                            "taken_time": time.time(),
                            "error": str(exc),
                        },
                    },
                )
            return False

    def _handle_loop_completion(self, task: asyncio.Task):
        try:
            if exception := task.exception():
                logger.error(f"{self.log_prefix} HeartFChatting: 主循环异常退出: {exception}")
                logger.error(traceback.format_exc())
            else:
                logger.info(f"{self.log_prefix} HeartFChatting: 主循环已退出")
        except asyncio.CancelledError:
            logger.info(f"{self.log_prefix} HeartFChatting: 聊天已结束")

    async def _trigger_expression_learning(self, messages: List["SessionMessage"]):
        if not messages:
            return

        self._expression_learner.add_messages(messages)
        if time.time() - self._last_extraction_time < self._min_extraction_interval:
            return
        if self._expression_learner.get_cache_size() < self._min_messages_for_extraction:
            return
        if not self._enable_expression_learning:
            return

        extraction_end_time = time.time()
        logger.info(
            f"聊天流 {self.session_name} 提取到 {len(messages)} 条消息，"
            f"时间窗口: {self._last_extraction_time:.2f} - {extraction_end_time:.2f}"
        )
        self._last_extraction_time = extraction_end_time

        try:
            jargon_miner = self._jargon_miner if self._enable_jargon_learning else None
            learnt_style = await self._expression_learner.learn(jargon_miner)
            if learnt_style:
                logger.info(f"{self.log_prefix} 表达学习完成")
            else:
                logger.debug(f"{self.log_prefix} 表达学习未获得有效结果")
        except Exception as exc:
            logger.error(f"{self.log_prefix} 表达学习失败: {exc}", exc_info=True)

    def _start_cycle(self) -> Tuple[Dict[str, float], str]:
        self._cycle_counter += 1
        self._current_cycle_detail = CycleDetail(cycle_id=self._cycle_counter)
        self._current_cycle_detail.thinking_id = f"tid{str(round(time.time(), 2))}"
        return self._current_cycle_detail.time_records, self._current_cycle_detail.thinking_id

    def _end_cycle(self, cycle_detail: Optional[CycleDetail], loop_info: Optional[Dict[str, Any]] = None):
        if cycle_detail is None:
            return None

        cycle_detail.loop_plan_info = (loop_info or {}).get("loop_plan_info")
        cycle_detail.loop_action_info = (loop_info or {}).get("loop_action_info")
        cycle_detail.end_time = time.time()
        self.history_loop.append(cycle_detail)

        timer_strings = [
            f"{name}: {duration:.2f}s"
            for name, duration in cycle_detail.time_records.items()
            if duration >= 0.1
        ]
        logger.info(
            f"{self.log_prefix} 第{cycle_detail.cycle_id} 个心流循环完成，"
            f"耗时: {cycle_detail.end_time - cycle_detail.start_time:.2f}s；"
            f"详细计时: {', '.join(timer_strings) if timer_strings else '无'}"
        )
        return cycle_detail

    async def _execute_action(
        self,
        action_planner_info: ActionPlannerInfo,
        chosen_action_plan_infos: List[ActionPlannerInfo],
        thinking_id: str,
        available_actions: Dict[str, ActionInfo],
        cycle_timers: Dict[str, float],
    ):
        try:
            with Timer(f"动作{action_planner_info.action_type}", cycle_timers):
                if action_planner_info.action_type == "no_reply":
                    reason = action_planner_info.reasoning or "选择不回复"
                    self._consecutive_no_reply_count += 1
                    await database_api.store_action_info(
                        chat_stream=self.chat_stream,
                        display_prompt=reason,
                        thinking_id=thinking_id,
                        action_data={},
                        action_name="no_reply",
                        action_reasoning=reason,
                    )
                    return {
                        "action_type": "no_reply",
                        "success": True,
                        "result": "选择不回复",
                        "loop_info": None,
                    }

                if action_planner_info.action_type == "reply":
                    self._consecutive_no_reply_count = 0
                    reason = action_planner_info.reasoning or ""
                    think_level = self._get_think_level(action_planner_info)
                    planner_reasoning = action_planner_info.action_reasoning or reason

                    record_replyer_action_temp(
                        chat_id=self.session_id,
                        reason=reason,
                        think_level=think_level,
                    )
                    await database_api.store_action_info(
                        chat_stream=self.chat_stream,
                        display_prompt=reason,
                        thinking_id=thinking_id,
                        action_data={},
                        action_name="reply",
                        action_reasoning=reason,
                    )

                    unknown_words, quote_message = self._extract_reply_metadata(action_planner_info)
                    success, llm_response = await generator_api.generate_reply(
                        chat_stream=self.chat_stream,
                        reply_message=action_planner_info.action_message,
                        available_actions=available_actions,
                        chosen_actions=chosen_action_plan_infos,
                        reply_reason=planner_reasoning,
                        unknown_words=unknown_words,
                        enable_tool=global_config.tool.enable_tool,
                        request_type="replyer",
                        from_plugin=False,
                        reply_time_point=action_planner_info.action_data.get("loop_start_time", time.time())
                        if action_planner_info.action_data
                        else time.time(),
                        think_level=think_level,
                    )
                    if not success or not llm_response or not llm_response.reply_set:
                        if action_planner_info.action_message:
                            logger.info(
                                f"对 {action_planner_info.action_message.processed_plain_text} 的回复生成失败"
                            )
                        else:
                            logger.info(f"{self.log_prefix} 回复生成失败")
                        return {
                            "action_type": "reply",
                            "success": False,
                            "result": "回复生成失败",
                            "loop_info": None,
                        }

                    loop_info, reply_text, _ = await self._send_and_store_reply(
                        response_set=llm_response.reply_set,
                        action_message=action_planner_info.action_message,  # type: ignore[arg-type]
                        cycle_timers=cycle_timers,
                        thinking_id=thinking_id,
                        actions=chosen_action_plan_infos,
                        selected_expressions=llm_response.selected_expressions,
                        quote_message=quote_message,
                    )
                    self.last_active_time = time.time()
                    return {
                        "action_type": "reply",
                        "success": True,
                        "result": reply_text,
                        "loop_info": loop_info,
                    }

                with Timer("动作执行", cycle_timers):
                    success, result = await self._handle_action(
                        action=action_planner_info.action_type,
                        action_reasoning=action_planner_info.action_reasoning or "",
                        action_data=action_planner_info.action_data or {},
                        cycle_timers=cycle_timers,
                        thinking_id=thinking_id,
                        action_message=action_planner_info.action_message,
                    )
                if success:
                    self.last_active_time = time.time()
                return {
                    "action_type": action_planner_info.action_type,
                    "success": success,
                    "result": result,
                    "loop_info": None,
                }
        except Exception as exc:
            logger.error(f"{self.log_prefix} 执行动作时出错: {exc}", exc_info=True)
            return {
                "action_type": action_planner_info.action_type,
                "success": False,
                "result": "",
                "loop_info": None,
                "error": str(exc),
            }

    async def _handle_action(
        self,
        action: str,
        action_reasoning: str,
        action_data: dict,
        cycle_timers: Dict[str, float],
        thinking_id: str,
        action_message: Optional["SessionMessage"] = None,
    ) -> Tuple[bool, str]:
        try:
            action_handler = self.action_manager.create_action(
                action_name=action,
                action_data=action_data,
                action_reasoning=action_reasoning,
                cycle_timers=cycle_timers,
                thinking_id=thinking_id,
                chat_stream=self.chat_stream,
                log_prefix=self.log_prefix,
                action_message=action_message,
            )
            if not action_handler:
                logger.warning(f"{self.log_prefix} 未能创建动作处理器: {action}")
                return False, ""

            success, action_text = await action_handler.execute()
            return success, action_text
        except Exception as exc:
            logger.error(f"{self.log_prefix} 处理动作 {action} 时出错: {exc}", exc_info=True)
            return False, ""

    async def _send_and_store_reply(
        self,
        response_set: MessageSequence,
        action_message: "SessionMessage",
        cycle_timers: Dict[str, float],
        thinking_id: str,
        actions: List[ActionPlannerInfo],
        selected_expressions: Optional[List[int]] = None,
        quote_message: Optional[bool] = None,
    ) -> Tuple[Dict[str, Any], str, Dict[str, float]]:
        with Timer("回复发送", cycle_timers):
            reply_text = await self._send_response(
                reply_set=response_set,
                message_data=action_message,
                selected_expressions=selected_expressions,
                quote_message=quote_message,
            )

        platform = action_message.platform or getattr(self.chat_stream, "platform", "unknown")
        person = Person(platform=platform, user_id=action_message.message_info.user_info.user_id)
        action_prompt_display = f"你对{person.person_name}进行了回复：{reply_text}"
        await database_api.store_action_info(
            chat_stream=self.chat_stream,
            display_prompt=action_prompt_display,
            thinking_id=thinking_id,
            action_data={"reply_text": reply_text},
            action_name="reply",
        )

        loop_info: Dict[str, Any] = {
            "loop_plan_info": {
                "action_result": actions,
            },
            "loop_action_info": {
                "action_taken": True,
                "reply_text": reply_text,
                "command": "",
                "taken_time": time.time(),
            },
        }
        return loop_info, reply_text, cycle_timers

    async def _send_response(
        self,
        reply_set: MessageSequence,
        message_data: "SessionMessage",
        selected_expressions: Optional[List[int]] = None,
        quote_message: Optional[bool] = None,
    ) -> str:
        if global_config.chat.llm_quote:
            need_reply = bool(quote_message)
        else:
            new_message_count = message_api.count_new_messages(
                chat_id=self.session_id,
                start_time=self.last_read_time,
                end_time=time.time(),
            )
            need_reply = new_message_count >= random.randint(2, 3) or time.time() - self.last_read_time > 90

        reply_text = ""
        first_replied = False
        for component in reply_set.components:
            if not isinstance(component, TextComponent):
                continue
            data = component.text
            if not first_replied:
                await send_api.text_to_stream(
                    text=data,
                    stream_id=self.session_id,
                    reply_message=message_data,
                    set_reply=need_reply,
                    typing=False,
                    selected_expressions=selected_expressions,
                )
                first_replied = True
            else:
                await send_api.text_to_stream(
                    text=data,
                    stream_id=self.session_id,
                    reply_message=message_data,
                    set_reply=False,
                    typing=True,
                    selected_expressions=selected_expressions,
                )
            reply_text += data
        return reply_text

    async def _build_planner_prompt_with_event(
        self,
        available_actions: Dict[str, ActionInfo],
        is_group_chat: bool,
        chat_target_info: Any,
        chat_content_block: str,
        message_id_list: List[Tuple[str, "SessionMessage"]],
    ) -> Tuple[Optional[str], Dict[str, ActionInfo]]:
        filtered_actions = self.action_planner._filter_actions_by_activation_type(available_actions, chat_content_block)
        prompt, _ = await self.action_planner.build_planner_prompt(
            is_group_chat=is_group_chat,
            chat_target_info=chat_target_info,
            current_available_actions=filtered_actions,
            chat_content_block=chat_content_block,
            message_id_list=message_id_list,
        )
        event_message = build_event_message(EventType.ON_PLAN, llm_prompt=prompt, stream_id=self.session_id)
        continue_flag, modified_message = await event_bus.emit(EventType.ON_PLAN, event_message)
        if not continue_flag:
            logger.info(f"{self.log_prefix} ON_PLAN 事件中止了本轮 HFC")
            return None, filtered_actions
        if modified_message and modified_message._modify_flags.modify_llm_prompt and modified_message.llm_prompt:
            prompt = modified_message.llm_prompt
        return prompt, filtered_actions

    def _ensure_force_reply_action(
        self,
        actions: List[ActionPlannerInfo],
        force_reply_message: Optional["SessionMessage"],
        available_actions: Dict[str, ActionInfo],
    ) -> List[ActionPlannerInfo]:
        if not force_reply_message:
            return actions

        has_reply_to_force_message = any(
            action.action_type == "reply"
            and action.action_message
            and action.action_message.message_id == force_reply_message.message_id
            for action in actions
        )
        if has_reply_to_force_message:
            return actions

        actions = [action for action in actions if action.action_type != "no_reply"]
        actions.insert(
            0,
            ActionPlannerInfo(
                action_type="reply",
                reasoning="用户提及了我，必须回复该消息",
                action_data={"loop_start_time": self.last_read_time},
                action_message=force_reply_message,
                available_actions=available_actions,
                action_reasoning=None,
            ),
        )
        logger.info(f"{self.log_prefix} 检测到强制回复消息，已补充 reply 动作")
        return actions

    def _log_plan(
        self,
        prompt: str,
        reasoning: str,
        llm_raw_output: Optional[str],
        llm_reasoning: Optional[str],
        llm_duration_ms: Optional[float],
        actions: List[ActionPlannerInfo],
    ) -> None:
        try:
            PlanReplyLogger.log_plan(
                chat_id=self.session_id,
                prompt=prompt,
                reasoning=reasoning,
                raw_output=llm_raw_output,
                raw_reasoning=llm_reasoning,
                actions=actions,
                timing={
                    "llm_duration_ms": round(llm_duration_ms, 2) if llm_duration_ms is not None else None,
                    "loop_start_time": self.last_read_time,
                },
                extra=None,
            )
        except Exception:
            logger.exception(f"{self.log_prefix} 记录 plan 日志失败")

    def _extract_reply_metadata(
        self,
        action_planner_info: ActionPlannerInfo,
    ) -> Tuple[Optional[List[str]], Optional[bool]]:
        unknown_words: Optional[List[str]] = None
        quote_message: Optional[bool] = None
        action_data = action_planner_info.action_data or {}

        raw_unknown_words = action_data.get("unknown_words")
        if isinstance(raw_unknown_words, list):
            cleaned_unknown_words = []
            for item in raw_unknown_words:
                if isinstance(item, str) and (cleaned_item := item.strip()):
                    cleaned_unknown_words.append(cleaned_item)
            if cleaned_unknown_words:
                unknown_words = cleaned_unknown_words

        raw_quote = action_data.get("quote")
        if isinstance(raw_quote, bool):
            quote_message = raw_quote
        elif isinstance(raw_quote, str):
            quote_message = raw_quote.lower() in {"true", "1", "yes"}
        elif isinstance(raw_quote, (int, float)):
            quote_message = bool(raw_quote)

        return unknown_words, quote_message

    def _get_think_level(self, action_planner_info: ActionPlannerInfo) -> int:
        think_mode = global_config.chat.think_mode
        if think_mode == "default":
            return 0
        if think_mode == "deep":
            return 1
        if think_mode == "dynamic":
            action_data = action_planner_info.action_data or {}
            return int(action_data.get("think_level", 1))
        return 0

    def _get_template_name(self) -> Optional[str]:
        if self.chat_stream.context:
            return self.chat_stream.context.template_name
        return None
