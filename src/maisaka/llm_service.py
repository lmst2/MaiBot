"""MaiSaka LLM 服务。

该模块基于主项目服务层封装 MaiSaka 所需的对话与工具调用接口。
"""

from base64 import b64decode
from dataclasses import dataclass
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

from src.common.data_models.mai_message_data_model import MaiMessage
from src.common.logger import get_logger
from src.common.prompt_i18n import load_prompt
from src.config.config import config_manager, global_config
from src.llm_models.model_client.base_client import BaseClient
from src.llm_models.payload_content.message import Message, MessageBuilder, RoleType
from src.llm_models.payload_content.tool_option import (
    ToolCall,
    ToolDefinitionInput,
    ToolOption,
    normalize_tool_options,
)
from src.common.data_models.llm_service_data_models import LLMGenerationOptions
from src.services.llm_service import LLMServiceClient

from . import config
from .config import console
from .builtin_tools import get_builtin_tools
from .message_adapter import (
    build_message,
    format_speaker_content,
    get_message_kind,
    get_message_role,
    get_message_text,
    get_tool_calls,
    to_llm_message,
)

logger = get_logger("maisaka_llm")

@dataclass(slots=True)
class ChatResponse:
    """LLM 对话循环单步响应。"""

    content: Optional[str]
    tool_calls: List[ToolCall]
    raw_message: MaiMessage


class MaiSakaLLMService:
    """MaiSaka LLM 服务 - 适配主项目 LLM 系统"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        chat_system_prompt: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 2048,
        enable_thinking: Optional[bool] = None,
    ) -> None:
        """初始化 MaiSaka LLM 服务。

        Args:
            api_key: 兼容旧接口保留的参数，当前不使用。
            base_url: 兼容旧接口保留的参数，当前不使用。
            model: 兼容旧接口保留的参数，当前不使用。
            chat_system_prompt: 可选的系统提示词覆盖值。
            temperature: 默认温度参数。
            max_tokens: 默认最大输出 token 数。
            enable_thinking: 是否启用思考模式。
        """
        del api_key, base_url, model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._enable_thinking = enable_thinking
        self._extra_tools: List[ToolOption] = []
        self._prompts_loaded = False
        self._prompt_load_lock = asyncio.Lock()

        # 初始化服务层 LLM 门面（按任务名实时解析配置，确保热重载生效）
        self._llm_tool_use = LLMServiceClient(task_name="tool_use", request_type="maisaka_tool_use")
        # 主对话也使用 planner 模型
        self._llm_planner = LLMServiceClient(task_name="planner", request_type="maisaka_planner")
        self._llm_chat = self._llm_planner
        self._llm_utils = self._llm_tool_use
        # 回复生成使用 replyer 模型
        self._llm_replyer = LLMServiceClient(task_name="replyer", request_type="maisaka_replyer")

        # 尝试修复数据库 schema（忽略错误）
        self._try_fix_database_schema()

        # 构建人设信息
        personality_prompt = self._build_personality_prompt()
        self._personality_prompt = personality_prompt

        # 提示词在真正调用 LLM 前异步懒加载，避免在已有事件循环中嵌套 run_until_complete
        if chat_system_prompt is None:
            self._chat_system_prompt = f"{personality_prompt}\n\n你是一个友好的 AI 助手。"
        else:
            self._chat_system_prompt = chat_system_prompt

        # 子模块提示词同样采用懒加载
        self._emotion_prompt: Optional[str] = None
        self._cognition_prompt: Optional[str] = None

    def get_current_model_name(self) -> str:
        """获取当前 Maisaka 对话主模型名称。

        Returns:
            str: 当前 planner 任务的首选模型名；未配置时返回 ``未配置``。
        """
        try:
            model_task_config = config_manager.get_model_config().model_task_config
            if model_task_config.planner.model_list:
                return model_task_config.planner.model_list[0]
        except Exception as exc:
            logger.warning(f"获取当前 Maisaka 模型名称失败: {exc}")
        return "未配置"

    def _try_fix_database_schema(self) -> None:
        """尝试修复数据库 schema。

        Returns:
            None: 该方法仅执行数据库修复副作用。
        """
        try:
            from src.common.database.database_client import get_db_session
            from sqlalchemy import text

            with get_db_session() as session:
                # 检查 model_api_provider_name 列是否存在
                result = session.execute(text("PRAGMA table_info(llm_usage)"))
                columns = [row[1] for row in result.fetchall()]

                if "model_api_provider_name" not in columns:
                    # 添加缺失的列
                    session.execute(text("ALTER TABLE llm_usage ADD COLUMN model_api_provider_name VARCHAR(255)"))
                    session.commit()
                    logger.info("数据库 schema 已修复：添加 model_api_provider_name 列")
        except Exception:
            # 静默忽略任何错误，不影响正常流程
            pass

    def _build_personality_prompt(self) -> str:
        """构建当前人设提示词。

        Returns:
            str: 最终用于系统提示词的人设描述。
        """
        try:
            bot_name = global_config.bot.nickname
            if global_config.bot.alias_names:
                bot_nickname = f",也有人叫你{','.join(global_config.bot.alias_names)}"
            else:
                bot_nickname = ""

            # 获取基础personality
            prompt_personality = global_config.personality.personality

            # 检查是否需要随机替换为状态（personality 本体）
            if (
                hasattr(global_config.personality, "states")
                and global_config.personality.states
                and hasattr(global_config.personality, "state_probability")
                and global_config.personality.state_probability > 0
                and random.random() < global_config.personality.state_probability
            ):
                # 随机选择一个状态替换personality
                selected_state = random.choice(global_config.personality.states)
                prompt_personality = selected_state

            prompt_personality = f"{prompt_personality};"
            return f"你的名字是{bot_name}{bot_nickname}，你{prompt_personality}"
        except Exception as e:
            logger.warning(f"构建人设信息失败: {e}")
            # 返回默认人设
            return "你的名字是麦麦，你是一个活泼可爱的AI助手。"

    def set_extra_tools(self, tools: List[ToolDefinitionInput]) -> None:
        """设置额外工具定义。

        Args:
            tools: 外部传入的工具定义列表，例如 MCP 暴露的 OpenAI-compatible 工具。
        """
        self._extra_tools = normalize_tool_options(tools) or []
        logger.info(f"已为 Maisaka 加载 {len(self._extra_tools)} 个额外工具")

    async def _ensure_prompts_loaded(self) -> None:
        """异步懒加载提示词。

        Returns:
            None: 该方法仅刷新内部提示词缓存。
        """
        if self._prompts_loaded:
            return

        async with self._prompt_load_lock:
            if self._prompts_loaded:
                return

            try:
                tools_section = ""
                if config.ENABLE_WRITE_FILE:
                    tools_section += "\n• write_file(filename, content) — 在 mai_files 目录下写入文件。"
                if config.ENABLE_READ_FILE:
                    tools_section += "\n• read_file(filename) — 读取 mai_files 目录下的文件内容。"
                if config.ENABLE_LIST_FILES:
                    tools_section += "\n• list_files() — 获取 mai_files 目录下所有文件的元信息列表。"
                self._chat_system_prompt = load_prompt(
                    "maidairy_chat",
                    file_tools_section=tools_section if tools_section else "",
                    bot_name=global_config.bot.nickname,
                    identity=self._personality_prompt,
                )
                logger.info(f"系统提示词已渲染，长度: {len(self._chat_system_prompt)}")
            except Exception as e:
                logger.error(f"加载系统提示词失败: {e}")
                self._chat_system_prompt = f"{self._personality_prompt}\n\n你是一个友好的 AI 助手。"

            try:
                self._emotion_prompt = load_prompt("maidairy_emotion")
                self._cognition_prompt = load_prompt("maidairy_cognition")
                logger.info("成功加载 MaiSaka 子模块提示词")
            except Exception as e:
                logger.warning(f"加载子模块提示词失败，将使用默认提示词: {e}")

            self._prompts_loaded = True

    @staticmethod
    def _get_role_badge_style(role: str) -> str:
        """为不同角色返回终端标签样式。

        Args:
            role: 消息角色名称。

        Returns:
            str: Rich 可识别的样式字符串。
        """
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
        """构建终端 ASCII 图片预览。

        Args:
            image_base64: 图片的 Base64 数据。

        Returns:
            Optional[str]: 可渲染的 ASCII 预览文本；失败时返回 `None`。
        """
        ascii_chars = " .:-=+*#%@"

        try:
            image_bytes = b64decode(image_base64)
            with PILImage.open(BytesIO(image_bytes)) as image:
                grayscale = image.convert("L")
                width, height = grayscale.size
                if width <= 0 or height <= 0:
                    return None

                preview_width = max(8, int(config.TERMINAL_IMAGE_PREVIEW_WIDTH))
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

    @staticmethod
    def _render_message_content(content: Any) -> object:
        """将消息内容转换为 Rich 可渲染对象。

        Args:
            content: 原始消息内容。

        Returns:
            object: Rich 可渲染对象。
        """
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
                        if config.TERMINAL_IMAGE_PREVIEW:
                            preview_text = MaiSakaLLMService._build_terminal_image_preview(image_base64)
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
        """将工具调用转换为 CLI 展示结构。

        Args:
            tool_call: 原始工具调用对象或字典。

        Returns:
            Dict[str, Any]: 统一后的展示字典。
        """
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
        """渲染单个工具调用面板。

        Args:
            tool_call: 原始工具调用对象或字典。
            index: 当前工具调用在父消息中的序号。
            parent_index: 父消息在消息列表中的序号。

        Returns:
            Panel: 可直接打印的工具调用面板。
        """
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
        """渲染主循环 Prompt 中的一条消息。

        Args:
            message: 原始消息对象或字典。
            index: 当前消息序号。

        Returns:
            Panel: 可直接打印的消息面板。
        """
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

    async def chat_loop_step(self, chat_history: List[MaiMessage]) -> ChatResponse:
        """执行主对话循环的一步。

        Args:
            chat_history: 当前对话历史。

        Returns:
            ChatResponse: 本轮对话生成结果。
        """
        await self._ensure_prompts_loaded()

        def message_factory(_client: BaseClient) -> List[Message]:
            """将 MaiSaka 对话历史转换为内部消息列表。

            Args:
                _client: 当前底层客户端实例。

            Returns:
                List[Message]: 规范化后的消息列表。
            """
            messages: List[Message] = []

            # 首先添加系统提示词
            system_msg = MessageBuilder().set_role(RoleType.System)
            system_msg.add_text_content(self._chat_system_prompt)
            messages.append(system_msg.build())

            # 然后添加对话历史
            for msg in chat_history:
                llm_message = to_llm_message(msg)
                if llm_message is not None:
                    messages.append(llm_message)

            return messages

        # 调用 LLM（使用带消息的接口）
        # 合并内置工具和额外工具，统一交给底层规范化流程处理。
        all_tools = [*get_builtin_tools(), *self._extra_tools]

        # 打印消息列表
        built_messages = message_factory(None)

        ordered_panels: List[Panel] = []
        for index, msg in enumerate(built_messages, start=1):
            ordered_panels.append(self._render_message_panel(msg, index))
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for tool_call_index, tool_call in enumerate(tool_calls, start=1):
                    ordered_panels.append(self._render_tool_call_panel(tool_call, tool_call_index, index))

        if config.SHOW_THINKING and ordered_panels:
            console.print(
                Panel(
                    Group(*ordered_panels),
                    title="MaiSaka LLM Request - chat_loop_step",
                    border_style="cyan",
                    padding=(0, 1),
                )
            )
            logger.info(f"chat_loop_step prompt display finished ({len(built_messages)} messages, {len(all_tools)} tools)")


        request_started_at = perf_counter()
        logger.info("chat_loop_step calling planner model generate_response_with_messages")
        generation_result = await self._llm_chat.generate_response_with_messages(
            message_factory=message_factory,
            options=LLMGenerationOptions(
                tool_options=all_tools if all_tools else None,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            ),
        )
        response = generation_result.response
        model = generation_result.model_name
        tool_calls = generation_result.tool_calls
        elapsed = perf_counter() - request_started_at
        logger.info(
            f"chat_loop_step planner model returned in {elapsed:.2f}s "
            f"(model={model}, tool_calls={len(tool_calls or [])}, response_len={len(response or '')})"
        )
        raw_message = build_message(
            role=RoleType.Assistant.value,
            content=response or "",
            source="assistant",
            tool_calls=tool_calls or None,
        )
        logger.info("chat_loop_step converted planner response into MaiMessage")

        return ChatResponse(
            content=response,
            tool_calls=tool_calls or [],
            raw_message=raw_message,
        )

    def _filter_for_api(self, chat_history: List[MaiMessage]) -> str:
        """将对话历史过滤为简单文本格式。

        Args:
            chat_history: 当前对话历史。

        Returns:
            str: 过滤后的文本上下文。
        """
        parts = []
        for msg in chat_history:
            role = get_message_role(msg)
            content = get_message_text(msg)

            # 跳过内部字段
            if get_message_kind(msg) == "perception" or role == RoleType.Tool.value:
                continue

            if role == RoleType.System.value:
                parts.append(f"System: {content}")
            elif role == RoleType.User.value:
                parts.append(f"User: {content}")
            elif role == RoleType.Assistant.value:
                # 处理工具调用
                tool_calls = get_tool_calls(msg)
                if tool_calls:
                    tool_desc = ", ".join([tc.func_name for tc in tool_calls if tc.func_name])
                    parts.append(f"Assistant (called tools: {tool_desc})")
                else:
                    parts.append(f"Assistant: {content}")

        return "\n\n".join(parts)

    def build_chat_context(self, user_text: str) -> List[MaiMessage]:
        """构建新的对话上下文。

        Args:
            user_text: 用户输入文本。

        Returns:
            List[MaiMessage]: 初始对话上下文消息列表。
        """
        return [
            build_message(
                role=RoleType.User.value,
                content=format_speaker_content(config.USER_NAME, user_text, datetime.now()),
                source="user",
            )
        ]

    # ──────── 分析模块（使用 utils 模型） ────────

    async def analyze_emotion(self, chat_history: List[MaiMessage]) -> str:
        """执行情绪分析。

        Args:
            chat_history: 当前对话历史。

        Returns:
            str: 情绪分析文本。
        """
        await self._ensure_prompts_loaded()
        filtered = [m for m in chat_history if get_message_kind(m) != "perception"]
        recent = filtered[-10:] if len(filtered) > 10 else filtered

        # 使用加载的系统提示词
        system_prompt = self._emotion_prompt or "请分析以下对话中用户的情绪状态和言语态度："

        prompt_parts = [f"{system_prompt}\n\n【对话内容】\n"]
        for msg in recent:
            role = get_message_role(msg)
            content = get_message_text(msg)
            if role == RoleType.User.value:
                prompt_parts.append(f"{config.USER_NAME}: {content}")
            elif role == RoleType.Assistant.value:
                prompt_parts.append(f"助手: {content}")

        prompt = "\n".join(prompt_parts)

        if config.SHOW_THINKING:
            print("\n" + "=" * 60)
            print("MaiSaka LLM Request - analyze_emotion:")
            print(f"  {prompt}")
            print("=" * 60 + "\n")

        try:
            generation_result = await self._llm_utils.generate_response(
                prompt=prompt,
                options=LLMGenerationOptions(temperature=0.3, max_tokens=512),
            )
            response = generation_result.response

            return response
        except Exception as e:
            logger.error(f"情绪分析 LLM 调用出错: {e}")
            return ""

    async def analyze_cognition(self, chat_history: List[MaiMessage]) -> str:
        """执行认知分析。

        Args:
            chat_history: 当前对话历史。

        Returns:
            str: 认知分析文本。
        """
        await self._ensure_prompts_loaded()
        filtered = [m for m in chat_history if get_message_kind(m) != "perception"]
        recent = filtered[-10:] if len(filtered) > 10 else filtered

        # 使用加载的系统提示词
        system_prompt = self._cognition_prompt or "请分析以下对话中用户的意图、认知状态和目的："

        prompt_parts = [f"{system_prompt}\n\n【对话内容】\n"]
        for msg in recent:
            role = get_message_role(msg)
            content = get_message_text(msg)
            if role == RoleType.User.value:
                prompt_parts.append(f"{config.USER_NAME}: {content}")
            elif role == RoleType.Assistant.value:
                prompt_parts.append(f"助手: {content}")

        prompt = "\n".join(prompt_parts)

        if config.SHOW_THINKING and config.SHOW_ANALYZE_COGNITION_PROMPT:
            print("\n" + "=" * 60)
            print("MaiSaka LLM Request - analyze_cognition:")
            print(f"  {prompt}")
            print("=" * 60 + "\n")

        try:
            generation_result = await self._llm_utils.generate_response(
                prompt=prompt,
                options=LLMGenerationOptions(temperature=0.3, max_tokens=512),
            )
            response = generation_result.response

            return response
        except Exception as e:
            logger.error(f"认知分析 LLM 调用出错: {e}")
            return ""

    async def _removed_analyze_timing(self, chat_history: List[MaiMessage], timing_info: str) -> str:
        """执行时间节奏分析。

        Args:
            chat_history: 当前对话历史。
            timing_info: 外部传入的时间信息摘要。

        Returns:
            str: 时间分析文本。
        """
        await self._ensure_prompts_loaded()
        filtered = [
            m
            for m in chat_history
            if get_message_kind(m) != "perception" and get_message_role(m) != RoleType.System.value
        ]

        # 使用加载的系统提示词
        system_prompt = self._timing_prompt or "请分析以下对话的时间节奏和用户状态："

        prompt_parts = [f"{system_prompt}\n\n【系统时间戳信息】\n{timing_info}\n\n【当前对话记录】\n"]
        for msg in filtered:
            role = get_message_role(msg)
            content = get_message_text(msg)
            if role == RoleType.User.value:
                prompt_parts.append(f"{config.USER_NAME}: {content}")
            elif role == RoleType.Assistant.value:
                prompt_parts.append(f"助手: {content}")

        prompt = "\n".join(prompt_parts)

        if False:
            print("\n" + "=" * 60)
            print("MaiSaka LLM Request - analyze_timing:")
            print(f"  {prompt}")
            print("=" * 60 + "\n")

        try:
            generation_result = await self._llm_utils.generate_response(
                prompt=prompt,
                options=LLMGenerationOptions(temperature=0.3, max_tokens=512),
            )
            response = generation_result.response

            return response
        except Exception as e:
            logger.error(f"时间分析 LLM 调用出错: {e}")
            return ""

    # ──────── 回复生成（使用 replyer 模型） ────────

    async def generate_reply(self, reason: str, chat_history: List[MaiMessage]) -> str:
        """生成最终回复文本。

        Args:
            reason: 当前轮次的内部想法或回复理由。
            chat_history: 当前对话历史。

        Returns:
            str: 最终回复文本。
        """
        await self._ensure_prompts_loaded()
        from datetime import datetime
        from .replyer import format_chat_history

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 格式化对话历史
        filtered_history = [
            msg
            for msg in chat_history
            if get_message_role(msg) != RoleType.System.value and get_message_kind(msg) != "perception"
        ]
        formatted_history = format_chat_history(filtered_history)

        # 获取回复提示词
        try:
            system_prompt = load_prompt(
                "maidairy_replyer",
                bot_name=global_config.bot.nickname,
                identity=self._personality_prompt,
                reply_style=global_config.personality.reply_style,
            )
        except Exception:
            system_prompt = "你是一个友好的 AI 助手，请根据用户的想法生成自然的回复。"

        user_prompt = (
            f"当前时间：{current_time}\n\n【聊天记录】\n{formatted_history}\n\n【你的想法】\n{reason}\n\n现在，你说："
        )

        messages = f"System: {system_prompt}\n\nUser: {user_prompt}"

        if config.SHOW_THINKING:
            print("\n" + "=" * 60)
            print("MaiSaka LLM Request - generate_reply:")
            print(f"  {messages}")
            print("=" * 60 + "\n")

        try:
            generation_result = await self._llm_replyer.generate_response(
                prompt=messages,
                options=LLMGenerationOptions(temperature=0.8, max_tokens=512),
            )
            response = generation_result.response
            return response.strip() if response else "..."
        except Exception as e:
            logger.error(f"回复生成 LLM 调用出错: {e}")
            return "..."
