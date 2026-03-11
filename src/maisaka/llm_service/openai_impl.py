"""
MaiSaka - OpenAI 兼容 LLM 服务实现
支持所有兼容 OpenAI Chat Completions 接口的服务商。
"""

import json
from typing import Callable, List, Optional

from openai import AsyncOpenAI

import asyncio

from .base import BaseLLMService, ChatResponse, ModelInfo, ToolCall
from .prompts import get_enabled_chat_tools
from .utils import format_chat_history, format_chat_history_for_eq, filter_for_api
from src.prompt.prompt_manager import prompt_manager
from knowledge import extract_category_ids_from_result


def _load_prompt_sync(name: str, **kwargs) -> str:
    """同步加载并渲染 prompt（用于非异步上下文）"""
    prompt = prompt_manager.get_prompt(name)
    for key, value in kwargs.items():
        prompt.add_context(key, value)
    # 在新事件循环中运行异步渲染
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(prompt_manager.render_prompt(prompt))
    finally:
        loop.close()


class OpenAILLMService(BaseLLMService):
    """
    基于 OpenAI 兼容 API 的 LLM 服务实现。
    支持所有兼容 OpenAI Chat Completions 接口的服务商。
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "gpt-4o",
        chat_system_prompt: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 2048,
        enable_thinking: Optional[bool] = None,
    ):
        """
        Args:
            api_key:              API 密钥
            base_url:             API 基地址 (默认 OpenAI 官方)
            model:                模型名称
            chat_system_prompt:   自定义对话系统提示词 (为 None 则使用默认)
            temperature:          生成温度
            max_tokens:           最大输出 token 数
            enable_thinking:      是否启用思考模式 (True/False/None)
        """
        self._base_url = base_url or "https://api.openai.com/v1"
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._enable_thinking = enable_thinking

        # 如果没有提供自定义提示词，则根据配置动态构建
        if chat_system_prompt is None:
            from config import ENABLE_WRITE_FILE, ENABLE_READ_FILE, ENABLE_LIST_FILES, ENABLE_QQ_TOOLS

            # 构建文件工具说明
            file_tools_parts = []
            if ENABLE_WRITE_FILE:
                file_tools_parts.append("• write_file(filename, content) — 在 mai_files 目录下写入文件，支持任意格式。")
            if ENABLE_READ_FILE:
                file_tools_parts.append("• read_file(filename) — 读取 mai_files 目录下的文件内容。")
            if ENABLE_LIST_FILES:
                file_tools_parts.append("• list_files() — 获取 mai_files 目录下所有文件的元信息列表。")

            # 构建QQ工具说明
            qq_tools_parts = []
            if ENABLE_QQ_TOOLS:
                qq_tools_parts.append("• get_qq_chat_info(chat, limit) — 获取指定 QQ 聊天的聊天记录。")
                qq_tools_parts.append("• send_info(chat, message) — 发送消息到指定的 QQ 聊天。")
                qq_tools_parts.append("• list_qq_chats() — 获取所有可用的 QQ 聊天列表。")

            # 合并所有工具说明
            tools_parts = []
            if file_tools_parts:
                tools_parts.extend(file_tools_parts)
            if qq_tools_parts:
                tools_parts.extend(qq_tools_parts)

            # 如果有任何工具启用，添加前缀空行
            if tools_parts:
                tools_section = "\n" + "\n".join(tools_parts) + "\n"
            else:
                tools_section = ""

            # 加载提示词模板并注入工具部分
            self._chat_system_prompt = _load_prompt_sync("maidairy_chat", file_tools_section=tools_section)
        else:
            self._chat_system_prompt = chat_system_prompt

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=self._base_url,
        )
        self._debug_callback: Optional[Callable] = None
        self._extra_tools: List[dict] = []  # MCP 等外部工具

    def set_extra_tools(self, tools: List[dict]) -> None:
        """设置额外的工具定义（如 MCP 工具），与内置工具合并使用。"""
        self._extra_tools = list(tools)

    def set_debug_callback(self, callback: Callable[[str, list, Optional[list], Optional[dict]], None]):
        """
        设置调试回调，每次 LLM 调用时触发（调用前和响应后）。

        callback(label, messages, tools, response) — tools 和 response 可为 None。
        """
        self._debug_callback = callback

    async def _call_llm(self, label: str, messages: list, tools: Optional[list] = None, **kwargs):
        """统一 LLM 调用入口：触发 debug 回调后调用 API。"""
        if self._debug_callback:
            try:
                self._debug_callback(label, messages, tools)
            except Exception:
                pass

        create_kwargs = {"model": self._model, "messages": messages, **kwargs}
        if tools:
            create_kwargs["tools"] = tools

        response = await self._client.chat.completions.create(**create_kwargs)

        # 发送响应结果到调试窗口
        if self._debug_callback:
            try:
                # 转换 tool_calls 为可序列化的格式
                tool_calls_list = []
                if response.choices[0].message.tool_calls:
                    for tc in response.choices[0].message.tool_calls:
                        tool_calls_list.append({
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        })

                resp_dict = {
                    "content": response.choices[0].message.content,
                    "tool_calls": tool_calls_list,
                }
                self._debug_callback(label, messages, tools, resp_dict)
            except Exception:
                pass

        return response

    def _build_extra_body(self) -> dict:
        """构建 extra_body 参数（如 enable_thinking）。"""
        extra_body = {}
        if self._enable_thinking is not None:
            extra_body["enable_thinking"] = self._enable_thinking
        return extra_body

    def _parse_tool_calls(self, msg) -> List[ToolCall]:
        """从 API 响应消息中解析工具调用列表。"""
        tool_calls: List[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))
        return tool_calls

    def _build_raw_message(self, msg) -> dict:
        """从 API 响应消息构建可追加到对话历史的消息字典。"""
        raw_message: dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            raw_message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        # 确保 arguments 是有效的 JSON 字符串，空参数用 "{}"
                        "arguments": tc.function.arguments or "{}",
                    },
                }
                for tc in msg.tool_calls
            ]
        return raw_message

    # ──────── 接口实现 ────────

    async def chat_loop_step(self, chat_history: List[dict]) -> ChatResponse:
        """执行对话循环的一步，返回包含文本和/或工具调用的响应。"""
        extra_body = self._build_extra_body()

        # 延迟导入配置以避免循环导入
        from config import ENABLE_WRITE_FILE, ENABLE_READ_FILE, ENABLE_LIST_FILES, ENABLE_QQ_TOOLS

        # 获取根据配置启用的内置工具
        enabled_tools = get_enabled_chat_tools(
            enable_write_file=ENABLE_WRITE_FILE,
            enable_read_file=ENABLE_READ_FILE,
            enable_list_files=ENABLE_LIST_FILES,
            enable_qq_tools=ENABLE_QQ_TOOLS,
        )

        # 合并内置工具与 MCP 等外部工具
        all_tools = enabled_tools + self._extra_tools

        # 过滤内部字段（如 _type），只保留 API 需要的字段
        api_messages = filter_for_api(chat_history)

        response = await self._call_llm(
            "主 Agent 对话",
            api_messages,
            tools=all_tools,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            **({"extra_body": extra_body} if extra_body else {}),
        )

        msg = response.choices[0].message
        return ChatResponse(
            content=msg.content,
            tool_calls=self._parse_tool_calls(msg),
            raw_message=self._build_raw_message(msg),
        )

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(model_name=self._model, base_url=self._base_url)

    # ──────── Timing 模块（含自我反思功能） ────────

    async def analyze_timing(
        self, chat_history: List[dict], timing_info: str,
    ) -> str:
        """Timing 模块（含自我反思功能）：分析对话的时间维度信息和进行自我反思。"""
        # 过滤掉感知消息和 system 消息
        filtered_history = [
            msg for msg in chat_history
            if msg.get("_type") != "perception" and msg.get("role") != "system"
        ]
        formatted = format_chat_history(filtered_history)
        timing_prompt = prompt_manager.get_prompt("maidairy_timing")
        timing_messages = [
            {"role": "system", "content": await prompt_manager.render_prompt(timing_prompt)},
            {
                "role": "user",
                "content": (
                    f"【系统时间戳信息】\n{timing_info}\n\n"
                    f"【当前对话记录】\n{formatted}"
                ),
            },
        ]
        extra_body = self._build_extra_body()

        response = await self._call_llm(
            "Timing 模块",
            timing_messages,
            temperature=0.3,
            max_tokens=512,
            **({"extra_body": extra_body} if extra_body else {}),
        )

        return response.choices[0].message.content or ""

    # ──────── 情商模块 (EQ Module) ────────

    async def analyze_emotion(self, chat_history: List[dict]) -> str:
        """情商模块：分析用户的情绪状态和言语态度。"""
        # 过滤掉感知消息（AI 的内部感知不需要再分析）
        filtered_history = [msg for msg in chat_history if msg.get("_type") != "perception"]
        # 获取最近几轮对话（约 8-10 条消息，约 3-5 轮）
        recent_messages = filtered_history[-10:] if len(filtered_history) > 10 else filtered_history
        # 使用情商模块专用格式化函数：只包含用户回复、助手思考、助手说
        formatted = format_chat_history_for_eq(recent_messages)

        emotion_prompt = prompt_manager.get_prompt("maidairy_emotion")
        eq_messages = [
            {"role": "system", "content": await prompt_manager.render_prompt(emotion_prompt)},
            {
                "role": "user",
                "content": f"以下是最近几轮对话记录，请分析其中用户的情绪状态和言语态度：\n\n{formatted}",
            },
        ]
        extra_body = self._build_extra_body()

        response = await self._call_llm(
            "情商模块 (EQ)",
            eq_messages,
            temperature=0.3,
            max_tokens=512,
            **({"extra_body": extra_body} if extra_body else {}),
        )

        return response.choices[0].message.content or ""

    # ──────── 认知模块 (Cognition Module) ────────

    async def analyze_cognition(self, chat_history: List[dict]) -> str:
        """认知模块：分析用户的意图、认知状态和目的。"""
        # 过滤掉感知消息（AI 的内部感知不需要再分析）
        filtered_history = [msg for msg in chat_history if msg.get("_type") != "perception"]
        # 获取最近几轮对话（约 8-10 条消息，约 3-5 轮）
        recent_messages = filtered_history[-10:] if len(filtered_history) > 10 else filtered_history
        # 使用情商模块专用格式化函数：只包含用户回复、助手思考、助手说
        formatted = format_chat_history_for_eq(recent_messages)

        cognition_prompt = prompt_manager.get_prompt("maidairy_cognition")
        cognition_messages = [
            {"role": "system", "content": await prompt_manager.render_prompt(cognition_prompt)},
            {
                "role": "user",
                "content": f"以下是最近几轮对话记录，请分析其中用户的意图、认知状态和目的：\n\n{formatted}",
            },
        ]
        extra_body = self._build_extra_body()

        response = await self._call_llm(
            "认知模块 (Cognition)",
            cognition_messages,
            temperature=0.3,
            max_tokens=512,
            **({"extra_body": extra_body} if extra_body else {}),
        )

        return response.choices[0].message.content or ""

    # ──────── 上下文总结模块 ────────

    async def summarize_context(self, context_messages: List[dict]) -> str:
        """上下文总结模块：对需要压缩的上下文进行总结。"""
        # 过滤掉 system 消息
        filtered_messages = [msg for msg in context_messages if msg.get("role") != "system"]
        formatted = format_chat_history(filtered_messages)

        summarize_prompt = prompt_manager.get_prompt("maidairy_context_summarize")
        summarize_messages = [
            {"role": "system", "content": await prompt_manager.render_prompt(summarize_prompt)},
            {
                "role": "user",
                "content": f"请对以下对话内容进行总结，以便存入记忆系统：\n\n{formatted}",
            },
        ]
        extra_body = self._build_extra_body()

        try:
            response = await self._call_llm(
                "上下文总结",
                summarize_messages,
                temperature=0.3,
                max_tokens=1024,
                **({"extra_body": extra_body} if extra_body else {}),
            )
            return response.choices[0].message.content or ""
        except Exception:
            # 总结失败时返回空字符串
            return ""

    # ──────── 了解模块 (Knowledge Module) ────────

    async def analyze_knowledge_categories(
        self, context_messages: List[dict], categories_summary: str
    ) -> List[str]:
        """
        了解模块-分类分析：分析对话内容涉及哪些个人特征分类。

        在上下文裁切时触发，分析需要提取哪些分类的个人特征信息。
        """
        from knowledge import format_context_for_memory

        context_text = format_context_for_memory(context_messages)
        if not context_text:
            return []

        # 加载分类分析 prompt
        category_prompt = prompt_manager.get_prompt("maidairy_knowledge_category")
        category_prompt.add_context("categories_summary", categories_summary)
        prompt = await prompt_manager.render_prompt(category_prompt)

        category_messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"请分析以下对话内容涉及哪些个人特征分类：\n\n{context_text}",
            },
        ]
        extra_body = self._build_extra_body()

        try:
            response = await self._call_llm(
                "了解模块-分类分析",
                category_messages,
                temperature=0.3,
                max_tokens=256,
                **({"extra_body": extra_body} if extra_body else {}),
            )
            result = response.choices[0].message.content or ""
            return extract_category_ids_from_result(result)
        except Exception:
            return []

    async def extract_knowledge_for_category(
        self, context_messages: List[dict], category_id: str, category_name: str
    ) -> str:
        """
        了解模块-内容提取：从对话中提取指定分类的个人特征信息。

        为每个分类创建 subAgent，提取相关的个人特征内容。
        """
        from knowledge import format_context_for_memory

        context_text = format_context_for_memory(context_messages)
        if not context_text:
            return ""

        # 加载内容提取 prompt
        extract_prompt = prompt_manager.get_prompt("maidairy_knowledge_extract")
        extract_prompt.add_context("category_name", category_name)
        prompt = await prompt_manager.render_prompt(extract_prompt)

        extract_messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"请从以下对话内容中提取与「{category_name}」相关的信息：\n\n{context_text}",
            },
        ]
        extra_body = self._build_extra_body()

        try:
            response = await self._call_llm(
                f"了解模块-{category_name}提取",
                extract_messages,
                temperature=0.3,
                max_tokens=512,
                **({"extra_body": extra_body} if extra_body else {}),
            )
            result = response.choices[0].message.content or ""

            # 检查是否表示"无"
            if "无" in result or not result.strip():
                return ""

            return result
        except Exception:
            return ""

    async def analyze_knowledge_need(
        self, chat_history: List[dict], categories_summary: str
    ) -> List[str]:
        """
        了解模块-需求分析：分析当前对话需要哪些个人特征信息。

        在每次对话前触发，分析需要检索哪些分类的了解内容。
        """
        # 过滤掉感知消息和 system 消息
        filtered_history = [
            msg for msg in chat_history
            if msg.get("_type") != "perception" and msg.get("role") != "system"
        ]
        # 获取最近几轮对话用于分析
        recent_messages = filtered_history[-10:] if len(filtered_history) > 10 else filtered_history
        formatted = format_chat_history(recent_messages)

        # 加载需求分析 prompt
        retrieve_prompt = prompt_manager.get_prompt("maidairy_knowledge_retrieve")
        retrieve_prompt.add_context("chat_context", formatted)
        retrieve_prompt.add_context("categories_summary", categories_summary)
        prompt = await prompt_manager.render_prompt(retrieve_prompt)

        need_messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": "请分析当前对话需要哪些个人特征信息。",
            },
        ]
        extra_body = self._build_extra_body()

        try:
            response = await self._call_llm(
                "了解模块-需求分析",
                need_messages,
                temperature=0.3,
                max_tokens=256,
                **({"extra_body": extra_body} if extra_body else {}),
            )
            result = response.choices[0].message.content or ""
            return extract_category_ids_from_result(result)
        except Exception:
            return []

    # ──────── 对话上下文构建 ────────

    def build_chat_context(self, user_text: str) -> List[dict]:
        """根据用户初始输入构建对话循环的初始上下文。"""
        return [
            {"role": "system", "content": self._chat_system_prompt},
            {"role": "user", "content": user_text},
        ]
