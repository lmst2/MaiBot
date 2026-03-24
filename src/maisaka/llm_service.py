"""
MaiSaka LLM 服务 - 使用主项目 LLM 系统
将主项目的 LLMRequest 适配为 MaiSaka 需要的接口
"""

from datetime import datetime

import asyncio
import random
from dataclasses import dataclass
from typing import Any, List, Optional

from rich.console import Group
from rich.panel import Panel
from rich.pretty import Pretty
from rich.text import Text

from src.common.data_models.mai_message_data_model import MaiMessage
from src.common.logger import get_logger
from src.common.prompt_i18n import load_prompt
from src.config.config import config_manager, global_config
from src.llm_models.payload_content.message import Message, MessageBuilder, RoleType
from src.llm_models.payload_content.tool_option import ToolCall, ToolOption
from src.llm_models.utils_model import LLMRequest

from . import config
from .config import console
from .builtin_tools import get_builtin_tools
from .message_adapter import (
    build_message,
    format_speaker_content,
    get_message_kind,
    get_message_role,
    get_message_text,
    get_tool_call_id,
    get_tool_calls,
    remove_last_perception,
    to_llm_message,
)

logger = get_logger("maisaka_llm")

@dataclass
class ChatResponse:
    """LLM 对话循环单步响应"""

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
    ):
        """
        初始化 LLM 服务

        参数仅为兼容性保留，实际使用主项目配置
        """
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._enable_thinking = enable_thinking
        self._extra_tools: List[dict] = []
        self._prompts_loaded = False
        self._prompt_load_lock = asyncio.Lock()

        # 获取主项目模型配置
        try:
            model_config = config_manager.get_model_config()
            self._model_configs = model_config.model_task_config
        except Exception:
            # 如果配置加载失败，使用默认配置
            from src.config.model_configs import ModelTaskConfig

            self._model_configs = ModelTaskConfig()
            logger.warning("无法加载主项目模型配置，使用默认配置")

        # 初始化 LLMRequest 实例（只使用 tool_use 和 replyer）
        self._llm_tool_use = LLMRequest(model_set=self._model_configs.tool_use, request_type="maisaka_tool_use")
        # 主对话也使用 tool_use 模型（因为需要工具调用支持）
        self._llm_planner = LLMRequest(model_set=self._model_configs.planner, request_type="maisaka_planner")
        self._llm_chat = self._llm_planner
        self._llm_utils = self._llm_tool_use
        # 回复生成使用 replyer 模型
        self._llm_replyer = LLMRequest(model_set=self._model_configs.replyer, request_type="maisaka_replyer")

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

        self._model_name = (
            self._model_configs.planner.model_list[0] if self._model_configs.planner.model_list else "未配置"
        )
        # 子模块提示词同样采用懒加载
        self._emotion_prompt: Optional[str] = None
        self._cognition_prompt: Optional[str] = None

    def _try_fix_database_schema(self) -> None:
        """尝试修复数据库 schema，添加缺失的列"""
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
        """构建人设信息，参考 replyer 的做法"""
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

    def set_extra_tools(self, tools: List[dict]) -> None:
        """设置额外的工具定义（如 MCP 工具）"""
        self._extra_tools = list(tools)

    async def _ensure_prompts_loaded(self) -> None:
        """异步懒加载提示词，避免在运行中的事件循环里同步渲染 prompt。"""
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
        """为不同 role 返回不同的标签样式。"""
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
    def _render_message_content(content: Any) -> object:
        """把消息内容转成适合 Rich 输出的 renderable。"""
        if isinstance(content, str):
            return Text(content)

        if isinstance(content, list):
            parts: list[object] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(Text(item))
                    continue
                if isinstance(item, tuple) and len(item) == 2:
                    image_format, image_base64 = item
                    if isinstance(image_format, str) and isinstance(image_base64, str):
                        approx_size = max(0, len(image_base64) * 3 // 4)
                        size_text = f"{approx_size / 1024:.1f} KB" if approx_size >= 1024 else f"{approx_size} B"
                        parts.append(
                            Panel(
                                Text(f"image/{image_format}  {size_text}\nbase64 omitted", style="magenta"),
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
    def _format_tool_call_for_display(tool_call: Any) -> dict[str, Any]:
        """将 tool call 转成适合 CLI 展示的结构。"""
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
        """Render assistant tool calls as standalone cards."""
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
        """渲染主循环 prompt 中的一条消息。"""
        if isinstance(message, dict):
            raw_role = message.get("role", "unknown")
            content = message.get("content")
            tool_calls = message.get("tool_calls")
            tool_call_id = message.get("tool_call_id")
        else:
            raw_role = getattr(message, "role", "unknown")
            content = getattr(message, "content", None)
            tool_calls = getattr(message, "tool_calls", None)
            tool_call_id = getattr(message, "tool_call_id", None)

        role = raw_role.value if hasattr(raw_role, "value") else str(raw_role)
        title = Text.assemble(
            Text(f" {role.upper()} ", style=self._get_role_badge_style(role)),
            Text(f"  #{index}", style="muted"),
        )

        parts: list[object] = []
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

    @staticmethod
    def _tool_option_to_dict(tool: "ToolOption") -> dict:
        """将 ToolOption 对象转换为主项目期望的 dict 格式

        主项目的 _build_tool_options() 期望的格式:
        {
            "name": str,
            "description": str,
            "parameters": List[Tuple[name, ToolParamType, description, required, enum_values]]
        }
        """
        params = []
        if tool.params:
            for param in tool.params:
                params.append((param.name, param.param_type, param.description, param.required, param.enum_values))
        return {"name": tool.name, "description": tool.description, "parameters": params}

    async def chat_loop_step(self, chat_history: list[MaiMessage]) -> ChatResponse:
        """执行对话循环的一步 - 使用 tool_use 模型"""
        await self._ensure_prompts_loaded()

        def message_factory(client) -> list[Message]:
            """将 MaiSaka 的 chat_history 转换为主项目的 Message 格式"""
            messages: list[Message] = []

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
        # 合并内置工具和额外工具（将 ToolOption 对象转换为 dict）
        all_tools = [self._tool_option_to_dict(t) for t in get_builtin_tools()] + (
            self._extra_tools if self._extra_tools else []
        )

        # 打印消息列表
        built_messages = message_factory(None)

        ordered_panels: list[Panel] = []
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


        response, (reasoning, model, tool_calls) = await self._llm_chat.generate_response_with_message_async(
            message_factory=message_factory,
            tools=all_tools if all_tools else None,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        raw_message = build_message(
            role=RoleType.Assistant.value,
            content=response or "",
            source="assistant",
            tool_calls=tool_calls or None,
        )

        return ChatResponse(
            content=response,
            tool_calls=tool_calls or [],
            raw_message=raw_message,
        )

    def _filter_for_api(self, chat_history: list[MaiMessage]) -> str:
        """过滤对话历史为 API 格式"""
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

    def build_chat_context(self, user_text: str) -> list[MaiMessage]:
        """构建对话上下文"""
        return [
            build_message(
                role=RoleType.User.value,
                content=format_speaker_content(config.USER_NAME, user_text, datetime.now()),
                source="user",
            )
        ]

    # ──────── 分析模块（使用 utils 模型） ────────

    async def analyze_emotion(self, chat_history: list[MaiMessage]) -> str:
        """情绪分析 - 使用 utils 模型"""
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
            response, _ = await self._llm_utils.generate_response_async(
                prompt=prompt,
                temperature=0.3,
                max_tokens=512,
            )

            return response
        except Exception as e:
            logger.error(f"情绪分析 LLM 调用出错: {e}")
            return ""

    async def analyze_cognition(self, chat_history: list[MaiMessage]) -> str:
        """认知分析 - 使用 utils 模型"""
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
            response, _ = await self._llm_utils.generate_response_async(
                prompt=prompt,
                temperature=0.3,
                max_tokens=512,
            )

            return response
        except Exception as e:
            logger.error(f"认知分析 LLM 调用出错: {e}")
            return ""

    async def _removed_analyze_timing(self, chat_history: list[MaiMessage], timing_info: str) -> str:
        """时间分析 - 使用 utils 模型"""
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
            response, _ = await self._llm_utils.generate_response_async(
                prompt=prompt,
                temperature=0.3,
                max_tokens=512,
            )

            return response
        except Exception as e:
            logger.error(f"时间分析 LLM 调用出错: {e}")
            return ""

    # ──────── 回复生成（使用 replyer 模型） ────────

    async def generate_reply(self, reason: str, chat_history: list[MaiMessage]) -> str:
        """
        生成回复 - 使用 replyer 模型
        可供 Replyer 类直接调用
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
            system_prompt = load_prompt("maidairy_replyer")
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
            response, _ = await self._llm_replyer.generate_response_async(
                prompt=messages,
                temperature=0.8,
                max_tokens=512,
            )
            return response.strip() if response else "..."
        except Exception as e:
            logger.error(f"回复生成 LLM 调用出错: {e}")
            return "..."





