"""Maisaka 对话循环服务。"""

from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import List, Optional, Sequence

import asyncio
import json
import random

from pydantic import BaseModel, Field as PydanticField
from rich.console import Group
from rich.panel import Panel

from src.cli.console import console
from src.common.data_models.llm_service_data_models import LLMGenerationOptions
from src.common.data_models.message_component_data_model import MessageSequence, TextComponent
from src.common.logger import get_logger
from src.common.prompt_i18n import load_prompt
from src.config.config import global_config
from src.core.tooling import ToolRegistry, ToolSpec
from src.know_u.knowledge import extract_category_ids_from_result
from src.llm_models.model_client.base_client import BaseClient
from src.llm_models.payload_content.message import Message, MessageBuilder, RoleType
from src.llm_models.payload_content.resp_format import RespFormat, RespFormatType
from src.llm_models.payload_content.tool_option import ToolCall, ToolDefinitionInput, ToolOption, normalize_tool_options
from src.services.llm_service import LLMServiceClient

from .builtin_tools import get_builtin_tools
from .context_messages import AssistantMessage, LLMContextMessage, SessionBackedMessage
from .message_adapter import format_speaker_content
from .prompt_cli_renderer import PromptCLIVisualizer


@dataclass(slots=True)
class ChatResponse:
    """LLM 对话循环单步响应。"""

    content: Optional[str]
    tool_calls: List[ToolCall]
    raw_message: AssistantMessage
    selected_history_count: int
    prompt_tokens: int
    built_message_count: int
    completion_tokens: int
    total_tokens: int


class ToolFilterSelection(BaseModel):
    """工具筛选响应。"""

    selected_tool_names: list[str] = PydanticField(default_factory=list)
    """经过预筛后保留的候选工具名称列表。"""


logger = get_logger("maisaka_chat_loop")


class MaisakaChatLoopService:
    """负责 Maisaka 主对话循环、系统提示词和终端渲染。"""

    def __init__(
        self,
        chat_system_prompt: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 2048,
    ) -> None:
        """初始化 Maisaka 对话循环服务。

        Args:
            chat_system_prompt: 可选的系统提示词。
            temperature: 规划器温度参数。
            max_tokens: 规划器最大输出长度。
        """

        self._temperature = temperature
        self._max_tokens = max_tokens
        self._extra_tools: List[ToolOption] = []
        self._interrupt_flag: asyncio.Event | None = None
        self._tool_registry: ToolRegistry | None = None
        self._prompts_loaded = False
        self._prompt_load_lock = asyncio.Lock()
        self._personality_prompt = self._build_personality_prompt()
        if chat_system_prompt is None:
            self._chat_system_prompt = f"{self._personality_prompt}\n\nYou are a helpful AI assistant."
        else:
            self._chat_system_prompt = chat_system_prompt
        self._llm_chat = LLMServiceClient(task_name="planner", request_type="maisaka_planner")
        self._tool_filter_llm = LLMServiceClient(
            task_name=global_config.maisaka.tool_filter_task_name,
            request_type="maisaka_tool_filter",
        )

    @property
    def personality_prompt(self) -> str:
        """返回当前人格提示词。"""

        return self._personality_prompt

    def _build_personality_prompt(self) -> str:
        """构造人格提示词。"""

        try:
            bot_name = global_config.bot.nickname
            if global_config.bot.alias_names:
                bot_nickname = f", also known as {','.join(global_config.bot.alias_names)}"
            else:
                bot_nickname = ""

            prompt_personality = global_config.personality.personality
            if (
                hasattr(global_config.personality, "states")
                and global_config.personality.states
                and hasattr(global_config.personality, "state_probability")
                and global_config.personality.state_probability > 0
                and random.random() < global_config.personality.state_probability
            ):
                prompt_personality = random.choice(global_config.personality.states)

            return f"Your name is {bot_name}{bot_nickname}; persona: {prompt_personality};"
        except Exception:
            return "Your name is MaiMai; persona: lively and cute AI assistant."

    async def ensure_chat_prompt_loaded(self, tools_section: str = "") -> None:
        """确保主聊天提示词已经加载完成。

        Args:
            tools_section: 额外注入到提示词中的工具说明片段。
        """

        if self._prompts_loaded:
            return

        async with self._prompt_load_lock:
            if self._prompts_loaded:
                return

            try:
                self._chat_system_prompt = load_prompt(
                    "maisaka_chat",
                    file_tools_section=tools_section,
                    bot_name=global_config.bot.nickname,
                    identity=self._personality_prompt,
                )
            except Exception:
                self._chat_system_prompt = f"{self._personality_prompt}\n\nYou are a helpful AI assistant."

            self._prompts_loaded = True

    def set_extra_tools(self, tools: Sequence[ToolDefinitionInput]) -> None:
        """设置额外工具定义。

        Args:
            tools: 兼容旧接口的额外工具定义列表。
        """

        self._extra_tools = normalize_tool_options(list(tools)) or []

    def set_tool_registry(self, tool_registry: ToolRegistry | None) -> None:
        """设置统一工具注册表。

        Args:
            tool_registry: 统一工具注册表；传入 ``None`` 时退回旧工具列表模式。
        """

        self._tool_registry = tool_registry

    def set_interrupt_flag(self, interrupt_flag: asyncio.Event | None) -> None:
        """设置当前 planner 请求使用的中断标记。"""
        self._interrupt_flag = interrupt_flag

    def _build_request_messages(self, selected_history: List[LLMContextMessage]) -> List[Message]:
        """构造发给大模型的消息列表。

        Args:
            selected_history: 已选中的上下文消息列表。

        Returns:
            List[Message]: 发送给大模型的消息列表。
        """

        messages: List[Message] = []
        system_msg = MessageBuilder().set_role(RoleType.System)
        system_msg.add_text_content(self._chat_system_prompt)
        messages.append(system_msg.build())

        for msg in selected_history:
            llm_message = msg.to_llm_message()
            if llm_message is not None:
                messages.append(llm_message)

        return messages

    @staticmethod
    def _is_builtin_tool_spec(tool_spec: ToolSpec) -> bool:
        """判断一个工具是否属于默认内置工具。

        Args:
            tool_spec: 待判断的工具声明。

        Returns:
            bool: 是否为默认内置工具。
        """

        return tool_spec.provider_type == "builtin" or tool_spec.provider_name == "maisaka_builtin"

    @classmethod
    def _split_builtin_and_candidate_tools(
        cls,
        tool_specs: List[ToolSpec],
    ) -> tuple[List[ToolSpec], List[ToolSpec]]:
        """拆分内置工具与可筛选工具列表。

        Args:
            tool_specs: 当前全部工具声明。

        Returns:
            tuple[List[ToolSpec], List[ToolSpec]]: `(内置工具, 可筛选工具)`。
        """

        builtin_tool_specs: List[ToolSpec] = []
        candidate_tool_specs: List[ToolSpec] = []
        for tool_spec in tool_specs:
            if cls._is_builtin_tool_spec(tool_spec):
                builtin_tool_specs.append(tool_spec)
            else:
                candidate_tool_specs.append(tool_spec)
        return builtin_tool_specs, candidate_tool_specs

    @staticmethod
    def _truncate_tool_filter_text(text: str, max_length: int = 180) -> str:
        """截断工具筛选阶段展示的文本。

        Args:
            text: 原始文本。
            max_length: 最长保留字符数。

        Returns:
            str: 截断后的文本。
        """

        normalized_text = text.strip()
        if len(normalized_text) <= max_length:
            return normalized_text
        return f"{normalized_text[: max_length - 1]}…"

    def _build_tool_filter_prompt(
        self,
        selected_history: List[LLMContextMessage],
        candidate_tool_specs: List[ToolSpec],
        max_keep: int,
    ) -> str:
        """构造小模型工具预筛选提示词。

        Args:
            selected_history: 已选中的对话上下文。
            candidate_tool_specs: 非内置候选工具列表。
            max_keep: 最多保留的候选工具数量。

        Returns:
            str: 用于工具预筛的小模型提示词。
        """

        history_lines: List[str] = []
        for message in selected_history[-10:]:
            plain_text = message.processed_plain_text.strip()
            if not plain_text:
                continue
            history_lines.append(
                f"- {message.role}: {self._truncate_tool_filter_text(plain_text, max_length=200)}"
            )

        if history_lines:
            history_section = "\n".join(history_lines)
        else:
            history_section = "- 当前没有可用的对话上下文。"

        tool_lines = [
            f"- {tool_spec.name}: {tool_spec.brief_description.strip() or '无简要描述'}"
            for tool_spec in candidate_tool_specs
        ]
        tool_section = "\n".join(tool_lines) if tool_lines else "- 当前没有候选工具。"

        return (
            "你是 Maisaka 的工具预筛选器。\n"
            "你的任务是在正式进入 planner 前，根据当前情景从候选工具中挑出最可能马上会用到的工具。\n"
            "默认内置工具已经自动保留，不在候选列表中，你不需要再次选择它们。\n"
            "你只能参考工具的简要描述，不要假设未描述的隐藏能力。\n"
            f"最多保留 {max_keep} 个候选工具；如果都不合适，可以返回空数组。\n"
            "请严格返回 JSON 对象，格式为："
            '{"selected_tool_names":["工具名1","工具名2"]}\n\n'
            f"【最近对话】\n{history_section}\n\n"
            f"【候选工具（仅简要描述）】\n{tool_section}"
        )

    @staticmethod
    def _parse_tool_filter_response(
        response_text: str,
        candidate_tool_specs: List[ToolSpec],
        max_keep: int,
    ) -> List[ToolSpec] | None:
        """解析工具预筛选响应。

        Args:
            response_text: 小模型返回的原始文本。
            candidate_tool_specs: 非内置候选工具列表。
            max_keep: 最多保留的候选工具数量。

        Returns:
            List[ToolSpec] | None: 成功解析时返回筛选后的工具列表；解析失败时返回 ``None``。
        """

        normalized_response = response_text.strip()
        if not normalized_response:
            return None

        selected_tool_names: List[str]
        try:
            selected_tool_names = ToolFilterSelection.model_validate_json(normalized_response).selected_tool_names
        except Exception:
            try:
                parsed_payload = json.loads(normalized_response)
            except json.JSONDecodeError:
                return None

            if isinstance(parsed_payload, dict):
                raw_tool_names = parsed_payload.get("selected_tool_names", [])
            elif isinstance(parsed_payload, list):
                raw_tool_names = parsed_payload
            else:
                return None

            if not isinstance(raw_tool_names, list):
                return None

            selected_tool_names = []
            for item in raw_tool_names:
                normalized_name = str(item).strip()
                if normalized_name:
                    selected_tool_names.append(normalized_name)

        candidate_map = {tool_spec.name: tool_spec for tool_spec in candidate_tool_specs}
        filtered_tool_specs: List[ToolSpec] = []
        seen_names: set[str] = set()
        for tool_name in selected_tool_names:
            normalized_name = tool_name.strip()
            if not normalized_name or normalized_name in seen_names:
                continue
            tool_spec = candidate_map.get(normalized_name)
            if tool_spec is None:
                continue

            seen_names.add(normalized_name)
            filtered_tool_specs.append(tool_spec)
            if len(filtered_tool_specs) >= max_keep:
                break

        return filtered_tool_specs

    async def _filter_tool_specs_for_planner(
        self,
        selected_history: List[LLMContextMessage],
        tool_specs: List[ToolSpec],
    ) -> List[ToolSpec]:
        """在将工具交给 planner 前进行快速预筛选。

        Args:
            selected_history: 已选中的对话上下文。
            tool_specs: 当前全部可用工具声明。

        Returns:
            List[ToolSpec]: 最终交给 planner 的工具声明列表。
        """

        threshold = max(1, int(global_config.maisaka.tool_filter_threshold))
        max_keep = max(1, int(global_config.maisaka.tool_filter_max_keep))
        if len(tool_specs) <= threshold:
            return tool_specs

        builtin_tool_specs, candidate_tool_specs = self._split_builtin_and_candidate_tools(tool_specs)
        if not candidate_tool_specs:
            return tool_specs
        if len(candidate_tool_specs) <= max_keep:
            return [*builtin_tool_specs, *candidate_tool_specs]

        filter_prompt = self._build_tool_filter_prompt(selected_history, candidate_tool_specs, max_keep)
        logger.info(
            "工具预筛选开始: "
            f"总工具数={len(tool_specs)} "
            f"内置工具数={len(builtin_tool_specs)} "
            f"候选工具数={len(candidate_tool_specs)} "
            f"最多保留候选数={max_keep}"
        )

        try:
            generation_result = await self._tool_filter_llm.generate_response(
                prompt=filter_prompt,
                options=LLMGenerationOptions(
                    temperature=0.0,
                    max_tokens=256,
                    response_format=RespFormat(
                        format_type=RespFormatType.JSON_SCHEMA,
                        schema=ToolFilterSelection,
                    ),
                ),
            )
        except Exception as exc:
            logger.warning(f"工具预筛选失败，保留全部工具。错误={exc}")
            return tool_specs

        filtered_candidate_tool_specs = self._parse_tool_filter_response(
            generation_result.response or "",
            candidate_tool_specs,
            max_keep,
        )
        if filtered_candidate_tool_specs is None:
            logger.warning(
                "工具预筛选返回结果无法解析，保留全部工具。"
                f" 原始返回={generation_result.response or ''!r}"
            )
            return tool_specs

        filtered_tool_specs = [*builtin_tool_specs, *filtered_candidate_tool_specs]
        if not filtered_tool_specs:
            logger.warning("工具预筛选得到空结果，保留全部工具以避免主流程失去工具能力。")
            return tool_specs

        logger.info(
            "工具预筛选完成: "
            f"筛选前总数={len(tool_specs)} "
            f"筛选后总数={len(filtered_tool_specs)} "
            f"保留候选工具={[tool_spec.name for tool_spec in filtered_candidate_tool_specs]}"
        )
        return filtered_tool_specs

    async def analyze_knowledge_need(
        self,
        chat_history: List[LLMContextMessage],
        categories_summary: str,
    ) -> List[str]:
        """分析当前对话是否需要检索知识库分类。"""
        visible_history: List[str] = []
        for message in chat_history[-8:]:
            if not message.processed_plain_text:
                continue
            visible_history.append(f"{message.role}: {message.processed_plain_text}")

        if not visible_history or not categories_summary.strip():
            return []

        prompt = (
            "你需要判断当前对话是否需要查询知识库。\n"
            "请只返回最相关的分类编号，多个编号用空格分隔；如果完全不需要，返回 none。\n\n"
            f"【可用分类】\n{categories_summary}\n\n"
            f"【最近对话】\n{chr(10).join(visible_history)}"
        )

        try:
            generation_result = await self._llm_chat.generate_response(
                prompt=prompt,
                options=LLMGenerationOptions(
                    temperature=0.1,
                    max_tokens=64,
                ),
            )
        except Exception:
            return []

        return extract_category_ids_from_result(generation_result.response or "")

    async def chat_loop_step(self, chat_history: List[LLMContextMessage]) -> ChatResponse:
        """执行一轮 Maisaka 规划器请求。

        Args:
            chat_history: 当前对话历史。

        Returns:
            ChatResponse: 本轮规划器返回结果。
        """

        await self.ensure_chat_prompt_loaded()
        selected_history, selection_reason = self._select_llm_context_messages(chat_history)
        built_messages = self._build_request_messages(selected_history)

        def message_factory(_client: BaseClient) -> List[Message]:
            """返回当前轮次已经构建好的请求消息。

            Args:
                _client: 当前模型客户端；此处不依赖客户端能力。

            Returns:
                List[Message]: 已经构建好的消息列表。
            """

            del _client
            return built_messages

        all_tools: List[ToolDefinitionInput]
        if self._tool_registry is not None:
            tool_specs = await self._tool_registry.list_tools()
            filtered_tool_specs = await self._filter_tool_specs_for_planner(selected_history, tool_specs)
            all_tools = [tool_spec.to_llm_definition() for tool_spec in filtered_tool_specs]
        else:
            all_tools = [*get_builtin_tools(), *self._extra_tools]

        ordered_panels = PromptCLIVisualizer.build_prompt_panels(
            built_messages,
            image_display_mode=global_config.maisaka.terminal_image_display_mode,
        )

        if global_config.maisaka.show_thinking and ordered_panels:
            console.print(
                Panel(
                    Group(*ordered_panels),
                    title="MaiSaka 大模型请求 - 对话单步",
                    subtitle=selection_reason,
                    border_style="cyan",
                    padding=(0, 1),
                )
            )

        request_started_at = perf_counter()
        logger.info(
            "规划器请求开始: "
            f"已选上下文消息数={len(selected_history)} "
            f"大模型消息数={len(built_messages)} "
            f"工具数={len(all_tools)} "
            f"启用打断={self._interrupt_flag is not None}"
        )
        generation_result = await self._llm_chat.generate_response_with_messages(
            message_factory=message_factory,
            options=LLMGenerationOptions(
                tool_options=all_tools if all_tools else None,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                interrupt_flag=self._interrupt_flag,
            ),
        )
        request_elapsed = perf_counter() - request_started_at
        logger.info(f"规划器请求完成，耗时={request_elapsed:.3f} 秒")

        prompt_stats_text = PromptCLIVisualizer.build_prompt_stats_text(
            selected_history_count=len(selected_history),
            built_message_count=len(built_messages),
            prompt_tokens=generation_result.prompt_tokens,
            completion_tokens=generation_result.completion_tokens,
            total_tokens=generation_result.total_tokens,
        )
        logger.info(f"本轮Prompt统计: {prompt_stats_text}")

        tool_call_summaries = [
            {
                "调用编号": getattr(tool_call, "call_id", getattr(tool_call, "id", None)),
                "工具名": getattr(tool_call, "func_name", getattr(tool_call, "name", None)),
                "参数": getattr(tool_call, "args", getattr(tool_call, "arguments", None)),
            }
            for tool_call in (generation_result.tool_calls or [])
        ]
        logger.info(
            f"Maisaka 规划器返回结果: 内容={generation_result.response or ''!r} "
            f"工具调用={tool_call_summaries}"
        )

        raw_message = AssistantMessage(
            content=generation_result.response or "",
            timestamp=datetime.now(),
            tool_calls=generation_result.tool_calls or [],
        )
        return ChatResponse(
            content=generation_result.response,
            tool_calls=generation_result.tool_calls or [],
            raw_message=raw_message,
            selected_history_count=len(selected_history),
            prompt_tokens=generation_result.prompt_tokens,
            built_message_count=len(built_messages),
            completion_tokens=generation_result.completion_tokens,
            total_tokens=generation_result.total_tokens,
        )

    @staticmethod
    def _select_llm_context_messages(chat_history: List[LLMContextMessage]) -> tuple[List[LLMContextMessage], str]:
        """选择真正发送给 LLM 的上下文消息。

        Args:
            chat_history: 当前全部对话历史。

        Returns:
            tuple[List[LLMContextMessage], str]: `(已选上下文, 选择说明)`。
        """

        max_context_size = max(1, int(global_config.chat.max_context_size))
        selected_indices: List[int] = []
        counted_message_count = 0

        for index in range(len(chat_history) - 1, -1, -1):
            message = chat_history[index]
            if message.to_llm_message() is None:
                continue

            selected_indices.append(index)
            if message.count_in_context:
                counted_message_count += 1
                if counted_message_count >= max_context_size:
                    break

        if not selected_indices:
            return [], f"上下文判定：最近 {max_context_size} 条 user/assistant（当前 0 条）"

        selected_indices.reverse()
        selected_history = [chat_history[index] for index in selected_indices]
        return (
            selected_history,
            (
                f"上下文判定：最近 {max_context_size} 条 user/assistant；"
                f"展示并发送窗口内消息 {len(selected_history)} 条"
            ),
        )

    @staticmethod
    def build_chat_context(user_text: str) -> List[LLMContextMessage]:
        """根据用户输入构造最小对话上下文。

        Args:
            user_text: 用户输入文本。

        Returns:
            List[LLMContextMessage]: 构造好的上下文消息列表。
        """

        timestamp = datetime.now()
        visible_text = format_speaker_content(
            global_config.maisaka.user_name.strip() or "用户",
            user_text,
            timestamp,
        )
        planner_prefix = (
            f"[时间]{timestamp.strftime('%H:%M:%S')}\n"
            f"[用户]{global_config.maisaka.user_name.strip() or '用户'}\n"
            "[用户群昵称]\n"
            "[msg_id]\n"
            "[发言内容]"
        )
        return [
            SessionBackedMessage(
                raw_message=MessageSequence([TextComponent(f"{planner_prefix}{user_text}")]),
                visible_text=visible_text,
                timestamp=timestamp,
                source_kind="user",
            )
        ]
