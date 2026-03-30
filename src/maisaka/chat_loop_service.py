"""Maisaka 对话循环服务。"""

from base64 import b64decode
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence

import asyncio
import random

from PIL import Image as PILImage
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.pretty import Pretty
from rich.text import Text

from src.cli.console import console
from src.common.data_models.llm_service_data_models import LLMGenerationOptions
from src.common.data_models.message_component_data_model import MessageSequence, TextComponent
from src.common.logger import get_logger
from src.common.prompt_i18n import load_prompt
from src.config.config import global_config
from src.core.tooling import ToolRegistry
from src.know_u.knowledge import extract_category_ids_from_result
from src.llm_models.model_client.base_client import BaseClient
from src.llm_models.payload_content.message import Message, MessageBuilder, RoleType
from src.llm_models.payload_content.tool_option import ToolCall, ToolDefinitionInput, ToolOption, normalize_tool_options
from src.services.llm_service import LLMServiceClient

from .builtin_tools import get_builtin_tools
from .context_messages import AssistantMessage, LLMContextMessage, SessionBackedMessage
from .message_adapter import format_speaker_content


@dataclass(slots=True)
class ChatResponse:
    """LLM 对话循环单步响应。"""

    content: Optional[str]
    tool_calls: List[ToolCall]
    raw_message: AssistantMessage


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
                    "maidairy_chat",
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
        """构造发给大模型的消息列表。"""
        messages: List[Message] = []
        system_msg = MessageBuilder().set_role(RoleType.System)
        system_msg.add_text_content(self._chat_system_prompt)
        messages.append(system_msg.build())

        for msg in selected_history:
            llm_message = msg.to_llm_message()
            if llm_message is not None:
                messages.append(llm_message)

        return messages

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

    @staticmethod
    def _get_role_badge_style(role: str) -> str:
        if role == "system":
            return "bold white on blue"
        if role == "user":
            return "bold black on green"
        if role == "assistant":
            return "bold black on yellow"
        if role == "tool":
            return "bold white on magenta"
        return "bold white on bright_black"

    @staticmethod
    def _get_role_badge_label(role: str) -> str:
        if role == "system":
            return "系统"
        if role == "user":
            return "用户"
        if role == "assistant":
            return "助手"
        if role == "tool":
            return "工具"
        return "未知"

    @staticmethod
    def _build_terminal_image_preview(image_base64: str) -> Optional[str]:
        ascii_chars = " .:-=+*#%@"

        try:
            image_bytes = b64decode(image_base64)
            with PILImage.open(BytesIO(image_bytes)) as image:
                grayscale = image.convert("L")
                width, height = grayscale.size
                if width <= 0 or height <= 0:
                    return None

                preview_width = max(8, int(global_config.maisaka.terminal_image_preview_width))
                preview_height = max(1, int(height * (preview_width / width) * 0.5))
                resized = grayscale.resize((preview_width, preview_height))
                pixels = list(resized.tobytes())
        except Exception:
            return None

        rows: List[str] = []
        for row_index in range(preview_height):
            row_pixels = pixels[row_index * preview_width : (row_index + 1) * preview_width]
            row = "".join(ascii_chars[min(len(ascii_chars) - 1, pixel * len(ascii_chars) // 256)] for pixel in row_pixels)
            rows.append(row)

        return "\n".join(rows)

    @classmethod
    def _render_message_content(cls, content: Any) -> RenderableType:
        if isinstance(content, str):
            return Text(content)

        if isinstance(content, list):
            parts: List[RenderableType] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(Text(item))
                    continue
                if isinstance(item, tuple) and len(item) == 2:
                    image_format, image_base64 = item
                    if isinstance(image_format, str) and isinstance(image_base64, str):
                        approx_size = max(0, len(image_base64) * 3 // 4)
                        size_text = f"{approx_size / 1024:.1f} KB" if approx_size >= 1024 else f"{approx_size} B"
                        preview_parts: List[RenderableType] = [
                            Text(f"图片格式 image/{image_format}  {size_text}\nbase64 内容已省略", style="magenta")
                        ]
                        if global_config.maisaka.terminal_image_preview:
                            preview_text = cls._build_terminal_image_preview(image_base64)
                            if preview_text:
                                preview_parts.append(Text(preview_text, style="white"))
                        parts.append(
                            Panel(
                                Group(*preview_parts),
                                border_style="magenta",
                                padding=(0, 1),
                            )
                        )
                        continue
                if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(Text(item["text"]))
                else:
                    parts.append(Pretty(item, expand_all=True))
            return Group(*parts) if parts else Text("")

        if content is None:
            return Text("")

        return Pretty(content, expand_all=True)

    @staticmethod
    def _format_tool_call_for_display(tool_call: Any) -> Dict[str, Any]:
        if isinstance(tool_call, dict):
            function_info = tool_call.get("function", {})
            return {
                "id": tool_call.get("id"),
                "name": function_info.get("name", tool_call.get("name")),
                "arguments": function_info.get("arguments", tool_call.get("arguments")),
            }

        return {
            "id": getattr(tool_call, "call_id", getattr(tool_call, "id", None)),
            "name": getattr(tool_call, "func_name", getattr(tool_call, "name", None)),
            "arguments": getattr(tool_call, "args", getattr(tool_call, "arguments", None)),
        }

    def _render_tool_call_panel(self, tool_call: Any, index: int, parent_index: int) -> Panel:
        title = Text.assemble(
            Text(" 工具调用 ", style="bold white on magenta"),
            Text(f"  #{parent_index}.{index}", style="muted"),
        )
        return Panel(
            Pretty(self._format_tool_call_for_display(tool_call), expand_all=True),
            title=title,
            border_style="magenta",
            padding=(0, 1),
        )

    def _render_message_panel(self, message: Any, index: int) -> Panel:
        if isinstance(message, dict):
            raw_role = message.get("role", "unknown")
            content = message.get("content")
            tool_call_id = message.get("tool_call_id")
        else:
            raw_role = getattr(message, "role", "unknown")
            content = getattr(message, "content", None)
            tool_call_id = getattr(message, "tool_call_id", None)

        role = raw_role.value if isinstance(raw_role, RoleType) else str(raw_role)
        title = Text.assemble(
            Text(f" {self._get_role_badge_label(role)} ", style=self._get_role_badge_style(role)),
            Text(f"  #{index}", style="muted"),
        )

        parts: List[RenderableType] = []
        if content not in (None, "", []):
            parts.append(Text(" 消息 ", style="bold cyan"))
            parts.append(self._render_message_content(content))

        if tool_call_id:
            parts.append(
                Text.assemble(
                    Text(" 工具调用编号 ", style="bold magenta"),
                    Text(" "),
                    Text(str(tool_call_id), style="magenta"),
                )
            )

        if not parts:
            parts.append(Text("[空消息]", style="muted"))

        return Panel(
            Group(*parts),
            title=title,
            border_style="dim",
            padding=(0, 1),
        )

    async def chat_loop_step(self, chat_history: List[LLMContextMessage]) -> ChatResponse:
        """执行一轮 Maisaka 规划器请求。

        Args:
            chat_history: 当前对话历史。

        Returns:
            ChatResponse: 本轮规划器返回结果。
        """

        await self.ensure_chat_prompt_loaded()
        selected_history, selection_reason = self._select_llm_context_messages(chat_history)

        def message_factory(_client: BaseClient) -> List[Message]:
            del _client
            return self._build_request_messages(selected_history)

        all_tools: List[ToolDefinitionInput]
        if self._tool_registry is not None:
            all_tools = await self._tool_registry.get_llm_definitions()
        else:
            all_tools = [*get_builtin_tools(), *self._extra_tools]
        built_messages = self._build_request_messages(selected_history)

        ordered_panels: List[Panel] = []
        for index, msg in enumerate(built_messages, start=1):
            ordered_panels.append(self._render_message_panel(msg, index))
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for tool_call_index, tool_call in enumerate(tool_calls, start=1):
                    ordered_panels.append(self._render_tool_call_panel(tool_call, tool_call_index, index))

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
        )

    @staticmethod
    def _select_llm_context_messages(chat_history: List[LLMContextMessage]) -> tuple[List[LLMContextMessage], str]:
        """选择真正发送给 LLM 的上下文消息。"""
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
