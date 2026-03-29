from dataclasses import dataclass
from base64 import b64decode
from datetime import datetime
from io import BytesIO
from time import perf_counter
from typing import Any, Dict, List, Optional

import asyncio
import random

from PIL import Image as PILImage
from rich.console import Group
from rich.panel import Panel
from rich.pretty import Pretty
from rich.text import Text

from src.chat.message_receive.message import SessionMessage
from src.cli.console import console
from src.common.data_models.llm_service_data_models import LLMGenerationOptions
from src.common.logger import get_logger
from src.common.prompt_i18n import load_prompt
from src.config.config import global_config
from src.know_u.knowledge import extract_category_ids_from_result
from src.llm_models.model_client.base_client import BaseClient
from src.llm_models.payload_content.message import Message, MessageBuilder, RoleType
from src.llm_models.payload_content.tool_option import ToolCall, ToolDefinitionInput, ToolOption, normalize_tool_options
from src.services.llm_service import LLMServiceClient

from .builtin_tools import get_builtin_tools
from .message_adapter import (
    build_message,
    format_speaker_content,
    to_llm_message,
)


@dataclass(slots=True)
class ChatResponse:
    """LLM 对话循环单步响应。"""

    content: Optional[str]
    tool_calls: List[ToolCall]
    raw_message: SessionMessage


logger = get_logger("maisaka_chat_loop")


class MaisakaChatLoopService:
    """负责 Maisaka 主对话循环、系统提示词和终端渲染。"""

    def __init__(
        self,
        chat_system_prompt: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 2048,
    ) -> None:
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._extra_tools: List[ToolOption] = []
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
        return self._personality_prompt

    def _build_personality_prompt(self) -> str:
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

    def set_extra_tools(self, tools: List[ToolDefinitionInput]) -> None:
        self._extra_tools = normalize_tool_options(tools) or []

    async def analyze_knowledge_need(
        self,
        chat_history: List[SessionMessage],
        categories_summary: str,
    ) -> List[str]:
        """分析当前对话是否需要检索知识库分类。"""
        visible_history: List[str] = []
        for message in chat_history[-8:]:
            if not message.content:
                continue
            role = getattr(message, "role", "")
            visible_history.append(f"{role}: {message.content}")

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
                pixels = list(resized.getdata())
        except Exception:
            return None

        rows: List[str] = []
        for row_index in range(preview_height):
            row_pixels = pixels[row_index * preview_width : (row_index + 1) * preview_width]
            row = "".join(ascii_chars[min(len(ascii_chars) - 1, pixel * len(ascii_chars) // 256)] for pixel in row_pixels)
            rows.append(row)

        return "\n".join(rows)

    @classmethod
    def _render_message_content(cls, content: Any) -> object:
        if isinstance(content, str):
            return Text(content)

        if isinstance(content, list):
            parts: List[object] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(Text(item))
                    continue
                if isinstance(item, tuple) and len(item) == 2:
                    image_format, image_base64 = item
                    if isinstance(image_format, str) and isinstance(image_base64, str):
                        approx_size = max(0, len(image_base64) * 3 // 4)
                        size_text = f"{approx_size / 1024:.1f} KB" if approx_size >= 1024 else f"{approx_size} B"
                        preview_parts: List[object] = [
                            Text(f"image/{image_format}  {size_text}\nbase64 omitted", style="magenta")
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
            Text(" TOOL CALL ", style="bold white on magenta"),
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

        role = raw_role.value if hasattr(raw_role, "value") else str(raw_role)
        title = Text.assemble(
            Text(f" {role.upper()} ", style=self._get_role_badge_style(role)),
            Text(f"  #{index}", style="muted"),
        )

        parts: List[object] = []
        if content not in (None, "", []):
            parts.append(Text(" message ", style="bold cyan"))
            parts.append(self._render_message_content(content))

        if tool_call_id:
            parts.append(
                Text.assemble(
                    Text(" tool_call_id ", style="bold magenta"),
                    Text(" "),
                    Text(str(tool_call_id), style="magenta"),
                )
            )

        if not parts:
            parts.append(Text("[empty message]", style="muted"))

        return Panel(
            Group(*parts),
            title=title,
            border_style="dim",
            padding=(0, 1),
        )

    async def chat_loop_step(self, chat_history: List[SessionMessage]) -> ChatResponse:
        await self.ensure_chat_prompt_loaded()

        def message_factory(_client: BaseClient) -> List[Message]:
            messages: List[Message] = []
            system_msg = MessageBuilder().set_role(RoleType.System)
            system_msg.add_text_content(self._chat_system_prompt)
            messages.append(system_msg.build())

            for msg in chat_history:
                llm_message = to_llm_message(msg)
                if llm_message is not None:
                    messages.append(llm_message)

            return messages

        all_tools = [*get_builtin_tools(), *self._extra_tools]
        built_messages = message_factory(None)

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
                    title="MaiSaka LLM Request - chat_loop_step",
                    border_style="cyan",
                    padding=(0, 1),
                )
            )

        request_started_at = perf_counter()
        generation_result = await self._llm_chat.generate_response_with_messages(
            message_factory=message_factory,
            options=LLMGenerationOptions(
                tool_options=all_tools if all_tools else None,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            ),
        )
        _ = perf_counter() - request_started_at

        tool_call_summaries = [
            {
                "id": getattr(tool_call, "call_id", getattr(tool_call, "id", None)),
                "name": getattr(tool_call, "func_name", getattr(tool_call, "name", None)),
                "args": getattr(tool_call, "args", getattr(tool_call, "arguments", None)),
            }
            for tool_call in (generation_result.tool_calls or [])
        ]
        logger.info(
            f"Maisaka planner returned content={generation_result.response or ''!r} "
            f"tool_calls={tool_call_summaries}"
        )

        raw_message = build_message(
            role=RoleType.Assistant.value,
            content=generation_result.response or "",
            source="assistant",
            tool_calls=generation_result.tool_calls or None,
        )
        return ChatResponse(
            content=generation_result.response,
            tool_calls=generation_result.tool_calls or [],
            raw_message=raw_message,
        )

    @staticmethod
    def build_chat_context(user_text: str) -> List[SessionMessage]:
        return [
            build_message(
                role=RoleType.User.value,
                content=format_speaker_content(
                    global_config.maisaka.user_name.strip() or "用户",
                    user_text,
                    datetime.now(),
                ),
                source="user",
            )
        ]
