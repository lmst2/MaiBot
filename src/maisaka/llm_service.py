"""
MaiSaka LLM 服务 - 使用主项目 LLM 系统
将主项目的 LLMRequest 适配为 MaiSaka 需要的接口
"""

from datetime import datetime

import json
import random
from dataclasses import dataclass
from typing import Any, List, Literal, Optional

from rich.console import Group
from rich.panel import Panel
from rich.pretty import Pretty
from rich.text import Text

from src.common.logger import get_logger
from src.config.config import config_manager, global_config
from src.llm_models.payload_content.message import MessageBuilder, RoleType
from src.llm_models.payload_content.tool_option import ToolCall as ToolCallOption, ToolOption
from src.llm_models.utils_model import LLMRequest
from src.prompt.prompt_manager import prompt_manager
from . import config
from .config import console
from .builtin_tools import get_builtin_tools

logger = get_logger("maisaka_llm")

# ──────────────────── 消息类型 ────────────────────

MessageType = Literal["user", "assistant", "system", "perception"]

# 内部使用的字段前缀，用于标记不应发送给 API 的元数据
INTERNAL_FIELD_PREFIX = "_"

# 消息类型字段名
MSG_TYPE_FIELD = "_type"


@dataclass
class ToolCall:
    """工具调用信息"""

    id: str
    name: str
    arguments: dict


@dataclass
class ChatResponse:
    """LLM 对话循环单步响应"""

    content: Optional[str]
    tool_calls: List[ToolCall]
    raw_message: dict  # 可直接追加到对话历史的消息字典


# ──────────────────── 工具函数 ────────────────────


def build_message(role: str, content: str, msg_type: MessageType = "user", **kwargs) -> dict:
    """构建消息字典，包含消息类型标记。"""
    msg = {
        "role": role,
        "content": content,
        MSG_TYPE_FIELD: msg_type,
        "_time": datetime.now().strftime("%H:%M:%S"),
        **kwargs,
    }
    return msg


def remove_last_perception(messages: list[dict]) -> None:
    """移除最后一条感知消息（直接修改原列表）。"""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get(MSG_TYPE_FIELD) == "perception":
            messages.pop(i)
            break


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

        # 加载系统提示词
        if chat_system_prompt is None:
            try:
                chat_prompt = prompt_manager.get_prompt("maidairy_chat")
                logger.info("成功加载 maidairy_chat 提示词模板")
                tools_section = ""
                if config.ENABLE_WRITE_FILE:
                    tools_section += "\n• write_file(filename, content) — 在 mai_files 目录下写入文件。"
                if config.ENABLE_READ_FILE:
                    tools_section += "\n• read_file(filename) — 读取 mai_files 目录下的文件内容。"
                if config.ENABLE_LIST_FILES:
                    tools_section += "\n• list_files() — 获取 mai_files 目录下所有文件的元信息列表。"

                chat_prompt.add_context("file_tools_section", tools_section if tools_section else "")
                chat_prompt.add_context("identity", personality_prompt)
                import asyncio

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    self._chat_system_prompt = loop.run_until_complete(prompt_manager.render_prompt(chat_prompt))
                    logger.info(f"系统提示词已渲染，长度: {len(self._chat_system_prompt)}")
                finally:
                    loop.close()
            except Exception as e:
                logger.error(f"加载系统提示词失败: {e}")
                self._chat_system_prompt = f"{personality_prompt}\n\n你是一个友好的 AI 助手。"
        else:
            self._chat_system_prompt = chat_system_prompt

        self._model_name = (
            self._model_configs.planner.model_list[0] if self._model_configs.planner.model_list else "未配置"
        )


        # 加载子模块提示词
        self._emotion_prompt: Optional[str] = None
        self._cognition_prompt: Optional[str] = None
        self._timing_prompt: Optional[str] = None
        self._context_summarize_prompt: Optional[str] = None

        try:
            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                self._emotion_prompt = loop.run_until_complete(
                    prompt_manager.render_prompt(prompt_manager.get_prompt("maidairy_emotion"))
                )
                self._cognition_prompt = loop.run_until_complete(
                    prompt_manager.render_prompt(prompt_manager.get_prompt("maidairy_cognition"))
                )
                self._timing_prompt = loop.run_until_complete(
                    prompt_manager.render_prompt(prompt_manager.get_prompt("maidairy_timing"))
                )
                self._context_summarize_prompt = loop.run_until_complete(
                    prompt_manager.render_prompt(prompt_manager.get_prompt("maidairy_context_summarize"))
                )
                logger.info("成功加载 MaiSaka 子模块提示词")
            finally:
                loop.close()
        except Exception as e:
            logger.warning(f"加载子模块提示词失败，将使用默认提示词: {e}")

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

        if tool_calls:
            parts.append(Text(" tool_calls ", style="bold magenta"))
            parts.append(
                Pretty(
                    [self._format_tool_call_for_display(tool_call) for tool_call in tool_calls],
                    expand_all=True,
                )
            )

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

    async def chat_loop_step(self, chat_history: List[dict]) -> ChatResponse:
        """执行对话循环的一步 - 使用 tool_use 模型"""

        def message_factory(client) -> List:
            """将 MaiSaka 的 chat_history 转换为主项目的 Message 格式"""
            messages = []

            # 首先添加系统提示词
            system_msg = MessageBuilder().set_role(RoleType.System)
            system_msg.add_text_content(self._chat_system_prompt)
            messages.append(system_msg.build())

            # 然后添加对话历史
            for msg in chat_history:
                role = msg.get("role", "")
                content = msg.get("content", "")

                # 跳过内部字段类型的消息和系统消息（已经有系统提示词了）
                if role in ("perception", "system"):
                    continue

                # 映射角色类型
                if role == "user":
                    role_type = RoleType.User
                elif role == "assistant":
                    role_type = RoleType.Assistant
                elif role == "tool":
                    role_type = RoleType.Tool
                else:
                    continue

                builder = MessageBuilder().set_role(role_type)

                # 处理工具调用
                if role == "assistant" and "tool_calls" in msg:
                    # 转换 tool_calls 格式：从 MaiSaka 格式转为主项目格式
                    tool_calls_list = []
                    for tc in msg["tool_calls"]:
                        tc_func = tc.get("function", {})
                        # 主项目的 ToolCall: call_id, func_name, args
                        tool_calls_list.append(
                            ToolCallOption(
                                call_id=tc.get("id", ""),
                                func_name=tc_func.get("name", ""),
                                args=json.loads(tc_func.get("arguments", "{}")) if tc_func.get("arguments") else {},
                            )
                        )
                    builder.set_tool_calls(tool_calls_list)
                elif role == "tool" and "tool_call_id" in msg:
                    builder.add_tool_call(msg["tool_call_id"])

                # 添加文本内容
                if content:
                    builder.add_text_content(content)

                messages.append(builder.build())

            return messages

        # 调用 LLM（使用带消息的接口）
        # 合并内置工具和额外工具（将 ToolOption 对象转换为 dict）
        all_tools = [self._tool_option_to_dict(t) for t in get_builtin_tools()] + (
            self._extra_tools if self._extra_tools else []
        )

        # 打印消息列表
        built_messages = message_factory(None)

        # 将消息分为普通消息和 tool 消息
        non_tool_panels = []
        tool_panels = []

        for index, msg in enumerate(built_messages):
            panel = self._render_message_panel(msg, index + 1)
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)

            if role == "tool":
                tool_panels.append(panel)
            else:
                non_tool_panels.append(panel)

        # 先显示普通消息（group 在一个 panel 内）
        if non_tool_panels:
            console.print(
                Panel(
                    Group(*non_tool_panels),
                    title="MaiSaka LLM Request - chat_loop_step",
                    border_style="cyan",
                    padding=(0, 1),
                )
            )

        # tool 消息作为单独的块展示
        for panel in tool_panels:
            console.print(panel)

        response, (reasoning, model, tool_calls) = await self._llm_chat.generate_response_with_message_async(
            message_factory=message_factory,
            tools=all_tools if all_tools else None,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        # 转换 tool_calls 格式：从主项目格式转为 MaiSaka 格式
        converted_tool_calls = []
        if tool_calls:
            for tc in tool_calls:
                # 主项目的 ToolCall 有 call_id, func_name, args
                call_id = tc.call_id if hasattr(tc, "call_id") else ""
                func_name = tc.func_name if hasattr(tc, "func_name") else ""
                args = tc.args if hasattr(tc, "args") else {}

                converted_tool_calls.append(
                    ToolCall(
                        id=call_id,
                        name=func_name,
                        arguments=args,
                    )
                )

        # 构建原始消息格式（MaiSaka 风格）
        raw_message = {
            "role": "assistant",
            "content": response,
            "_time": datetime.now().strftime("%H:%M:%S"),
        }
        if converted_tool_calls:
            raw_message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in converted_tool_calls
            ]

        return ChatResponse(
            content=response,
            tool_calls=converted_tool_calls,
            raw_message=raw_message,
        )

    def _filter_for_api(self, chat_history: List[dict]) -> str:
        """过滤对话历史为 API 格式"""
        parts = []
        for msg in chat_history:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # 跳过内部字段
            if role in ("perception", "tool"):
                continue

            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                # 处理工具调用
                if "tool_calls" in msg:
                    tool_desc = ", ".join([tc.get("name", "") for tc in msg["tool_calls"]])
                    parts.append(f"Assistant (called tools: {tool_desc})")
                else:
                    parts.append(f"Assistant: {content}")

        return "\n\n".join(parts)

    def build_chat_context(self, user_text: str) -> List[dict]:
        """构建对话上下文"""
        return [
            {"role": "system", "content": self._chat_system_prompt},
            {"role": "user", "content": user_text},
        ]

    # ──────── 分析模块（使用 utils 模型） ────────

    async def analyze_emotion(self, chat_history: List[dict]) -> str:
        """情绪分析 - 使用 utils 模型"""
        filtered = [m for m in chat_history if m.get("_type") != "perception"]
        recent = filtered[-10:] if len(filtered) > 10 else filtered

        # 使用加载的系统提示词
        system_prompt = self._emotion_prompt or "请分析以下对话中用户的情绪状态和言语态度："

        prompt_parts = [f"{system_prompt}\n\n【对话内容】\n"]
        for msg in recent:
            if msg.get("role") == "user":
                prompt_parts.append(f"用户: {msg.get('content', '')}")
            elif msg.get("role") == "assistant":
                prompt_parts.append(f"助手: {msg.get('content', '')}")

        prompt = "\n".join(prompt_parts)

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

    async def analyze_cognition(self, chat_history: List[dict]) -> str:
        """认知分析 - 使用 utils 模型"""
        filtered = [m for m in chat_history if m.get("_type") != "perception"]
        recent = filtered[-10:] if len(filtered) > 10 else filtered

        # 使用加载的系统提示词
        system_prompt = self._cognition_prompt or "请分析以下对话中用户的意图、认知状态和目的："

        prompt_parts = [f"{system_prompt}\n\n【对话内容】\n"]
        for msg in recent:
            if msg.get("role") == "user":
                prompt_parts.append(f"用户: {msg.get('content', '')}")
            elif msg.get("role") == "assistant":
                prompt_parts.append(f"助手: {msg.get('content', '')}")

        prompt = "\n".join(prompt_parts)

        if config.SHOW_ANALYZE_COGNITION_PROMPT:
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

    async def analyze_timing(self, chat_history: List[dict], timing_info: str) -> str:
        """时间分析 - 使用 utils 模型"""
        filtered = [m for m in chat_history if m.get("_type") not in ("perception", "system")]

        # 使用加载的系统提示词
        system_prompt = self._timing_prompt or "请分析以下对话的时间节奏和用户状态："

        prompt_parts = [f"{system_prompt}\n\n【系统时间戳信息】\n{timing_info}\n\n【当前对话记录】\n"]
        for msg in filtered:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                prompt_parts.append(f"用户: {content}")
            elif role == "assistant":
                prompt_parts.append(f"助手: {content}")

        prompt = "\n".join(prompt_parts)

        if config.SHOW_ANALYZE_TIMING_PROMPT:
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

    async def summarize_context(self, context_messages: List[dict]) -> str:
        """上下文总结 - 使用 utils 模型"""
        filtered = [m for m in context_messages if m.get("role") != "system"]

        # 使用加载的系统提示词
        system_prompt = self._context_summarize_prompt or "请对以下对话内容进行总结："

        prompt_parts = [f"{system_prompt}\n\n【对话内容】\n"]
        for msg in filtered:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                prompt_parts.append(f"用户: {content}")
            elif role == "assistant":
                prompt_parts.append(f"助手: {content}")

        prompt = "\n".join(prompt_parts)

        print("\n" + "=" * 60)
        print("MaiSaka LLM Request - summarize_context:")
        print(f"  {prompt}")
        print("=" * 60 + "\n")

        try:
            response, _ = await self._llm_utils.generate_response_async(
                prompt=prompt,
                temperature=0.3,
                max_tokens=1024,
            )

            return response
        except Exception as e:
            logger.error(f"上下文总结 LLM 调用出错: {e}")
            return ""

    # ──────── 回复生成（使用 replyer 模型） ────────

    async def generate_reply(self, reason: str, chat_history: List[dict]) -> str:
        """
        生成回复 - 使用 replyer 模型
        可供 Replyer 类直接调用
        """
        from datetime import datetime
        from .replyer import format_chat_history

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 格式化对话历史
        filtered_history = [
            msg for msg in chat_history if msg.get("role") != "system" and msg.get("_type") != "perception"
        ]
        formatted_history = format_chat_history(filtered_history)

        # 获取回复提示词
        try:
            replyer_prompt = prompt_manager.get_prompt("maidairy_replyer")
            system_prompt = await prompt_manager.render_prompt(replyer_prompt)
        except Exception:
            system_prompt = "你是一个友好的 AI 助手，请根据用户的想法生成自然的回复。"

        user_prompt = (
            f"当前时间：{current_time}\n\n【聊天记录】\n{formatted_history}\n\n【你的想法】\n{reason}\n\n现在，你说："
        )

        messages = f"System: {system_prompt}\n\nUser: {user_prompt}"

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





