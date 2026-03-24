"""
MaiSaka - CLI 交互界面与对话引擎
BufferCLI 整合主循环、对话引擎、子代理管理。
"""

import os
import asyncio
from datetime import datetime
from typing import Optional

from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from rich import box

from .config import (
    console,
    ENABLE_EMOTION_MODULE,
    ENABLE_COGNITION_MODULE,
    ENABLE_TIMING_MODULE,
    ENABLE_KNOWLEDGE_MODULE,
    ENABLE_MCP,
)
from .input_reader import InputReader
from .knowledge import retrieve_relevant_knowledge, store_knowledge_from_context
from .knowledge_store import get_knowledge_store
from .llm_service import MaiSakaLLMService, build_message, remove_last_perception
from .mcp_client import MCPManager
from .timing import build_timing_info
from .tool_handlers import (
    ToolHandlerContext,
    handle_list_files,
    handle_mcp_tool,
    handle_read_file,
    handle_stop,
    handle_unknown_tool,
    handle_wait,
    handle_write_file,
)


class BufferCLI:
    """命令行交互界面"""

    def __init__(self):
        self.llm_service: Optional[MaiSakaLLMService] = None
        self._reader = InputReader()
        self._chat_history: Optional[list] = None  # 持久化的对话历史
        self._knowledge_store = get_knowledge_store()  # 了解存储实例

        # 显示了解存储统计
        knowledge_stats = self._knowledge_store.get_stats()
        if knowledge_stats["total_items"] > 0:
            console.print(f"[success][OK] 了解系统: {knowledge_stats['total_items']}条特征信息[/success]")
        else:
            console.print("[muted][OK] 了解系统: 已初始化 (暂无数据)[/muted]")
        # Timing 模块时间戳跟踪
        self._chat_start_time: Optional[datetime] = None
        self._last_user_input_time: Optional[datetime] = None
        self._last_assistant_response_time: Optional[datetime] = None
        self._user_input_times: list[datetime] = []  # 所有用户输入时间戳
        # MCP 管理器（异步初始化，在 run() 中完成）
        self._mcp_manager: Optional[MCPManager] = None
        self._init_llm()

    def _init_llm(self):
        """初始化 LLM 服务 - 使用主项目配置系统"""
        thinking_env = os.getenv("ENABLE_THINKING", "").strip().lower()
        enable_thinking: Optional[bool] = True if thinking_env == "true" else False if thinking_env == "false" else None

        # MaiSakaLLMService 现在使用主项目的配置系统
        # 参数仅为兼容性保留，实际从 config_manager 读取配置
        self.llm_service = MaiSakaLLMService(
            api_key="",
            base_url=None,
            model="",
            enable_thinking=enable_thinking,
        )

        # 获取实际使用的模型名称
        model_name = self.llm_service._model_name
        console.print(f"[success][OK] LLM 服务已初始化[/success] [muted](模型: {model_name})[/muted]")

    def _build_tool_context(self) -> ToolHandlerContext:
        """构建工具处理器所需的上下文。"""
        ctx = ToolHandlerContext(
            llm_service=self.llm_service,
            reader=self._reader,
            user_input_times=self._user_input_times,
        )
        ctx.last_user_input_time = self._last_user_input_time
        return ctx

    def _show_banner(self):
        """显示欢迎横幅"""
        banner = Text()
        banner.append("MaiSaka", style="bold cyan")
        banner.append(" v2.0\n", style="muted")
        banner.append("直接输入文字开始对话 | Ctrl+C 退出", style="muted")

        console.print(Panel(banner, box=box.DOUBLE_EDGE, border_style="cyan", padding=(1, 2)))
        console.print()

    # ──────── 上下文管理 ────────

    def _get_safe_removal_indices(self, chat_history: list, count: int) -> list[int]:
        """
        获取可以安全删除的消息索引。

        确保 tool_calls 和 tool 响应消息成对删除，避免破坏 API 要求的配对关系。
        只删除完整的消息块（user/assistant + 可选的 tool 响应序列）。

        保留最后 3 条非 tool 消息，避免删除可能还在处理中的内容。

        Returns:
            可以安全删除的消息索引列表（从后往前排序）
        """
        indices_to_remove = []
        removed_count = 0
        i = 0

        # 计算保留的消息数量（最后 3 条非 tool 消息）
        safe_zone_count = 3
        non_tool_count = 0
        for msg in reversed(chat_history):
            if msg.get("role") != "tool":
                non_tool_count += 1
            if non_tool_count >= safe_zone_count:
                break

        # 只处理前 (len - non_tool_count) 条消息
        max_process_index = len(chat_history) - non_tool_count

        while i < max_process_index and removed_count < count:
            msg = chat_history[i]
            role = msg.get("role", "")

            # 跳过 role=tool 的消息（它们会被对应的 assistant 消息一起处理）
            if role == "tool":
                i += 1
                continue

            # 检查这是否是一个带 tool_calls 的 assistant 消息
            if role == "assistant" and "tool_calls" in msg:
                # 收集这个 assistant 消息及其后续的 tool 响应消息
                block_indices = [i]
                j = i + 1
                while j < len(chat_history):
                    next_msg = chat_history[j]
                    if next_msg.get("role") == "tool":
                        block_indices.append(j)
                        j += 1
                    else:
                        break
                indices_to_remove.extend(block_indices)
                removed_count += 1
                i = j
            elif role in ["user", "assistant"]:
                # 普通消息，可以直接删除
                indices_to_remove.append(i)
                removed_count += 1
                i += 1
            else:
                i += 1

        # 从后往前排序，避免索引问题
        return sorted(indices_to_remove, reverse=True)

    async def _manage_context_length(self, chat_history: list) -> None:
        """
        上下文管理：当对话历史过长时进行压缩。

        当达到 20 条上下文时：
        1. 移除最早 10 条上下文
        2. 对这 10 条内容进行 LLM 总结
        3. 将总结后的内容存入记忆
        """
        CONTEXT_LIMIT = 20
        COMPRESS_COUNT = 10

        # 计算实际消息数量（排除 role=tool 的工具返回消息）
        actual_messages = [m for m in chat_history if m.get("role") != "tool"]

        if len(actual_messages) >= CONTEXT_LIMIT:
            # 获取安全删除的索引
            indices_to_remove = self._get_safe_removal_indices(chat_history, COMPRESS_COUNT)

            if indices_to_remove:
                # 收集要总结的消息（在删除前）
                to_compress = []
                for i in sorted(indices_to_remove):
                    if 0 <= i < len(chat_history):
                        to_compress.append(chat_history[i])

                if to_compress:
                    # 总结上下文
                    try:
                        console.print("[accent]🧠 上下文过长，正在压缩并存入记忆...[/accent]")
                        summary = await self.llm_service.summarize_context(to_compress)

                        # 存储了解信息（如果启用）
                        if ENABLE_KNOWLEDGE_MODULE:
                            try:
                                knowledge_count = await store_knowledge_from_context(
                                    self.llm_service,
                                    to_compress,
                                    store_result_callback=lambda cat_id, cat_name, content: console.print(
                                        f"[muted]  [OK] 存储了解信息: {cat_name}[/muted]"
                                    ),
                                )
                                if knowledge_count > 0:
                                    console.print(f"[success][OK] 了解模块: 存储{knowledge_count}条特征信息[/success]")
                            except Exception as e:
                                console.print(f"[warning]了解存储失败: {e}[/warning]")
                        if summary:
                            # 存入记忆
                            # 显示压缩结果
                            console.print(
                                Panel(
                                    Markdown(summary),
                                    title="📝 上下文已压缩",
                                    border_style="green",
                                    padding=(0, 1),
                                    style="dim",
                                )
                            )
                    except Exception as e:
                        console.print(f"[warning]上下文总结失败: {e}[/warning]")

                # 从后往前删除
                for i in indices_to_remove:
                    if 0 <= i < len(chat_history):
                        chat_history.pop(i)

                # 清理"孤儿" tool 消息（没有对应 tool_calls 的 tool 消息）
                valid_tool_call_ids = set()
                for msg in chat_history:
                    if msg.get("role") == "assistant" and "tool_calls" in msg:
                        for tool_call in msg["tool_calls"]:
                            valid_tool_call_ids.add(tool_call.get("id", ""))

                # 删除无效的 tool 消息（从后往前）
                i = len(chat_history) - 1
                while i >= 0:
                    msg = chat_history[i]
                    if msg.get("role") == "tool":
                        tool_call_id = msg.get("tool_call_id", "")
                        if tool_call_id not in valid_tool_call_ids:
                            chat_history.pop(i)
                    i -= 1

    # ──────── LLM 循环架构 ────────

    async def _start_chat(self, user_text: str):
        """接收用户输入并启动/继续 LLM 对话循环"""
        if not self.llm_service:
            console.print("[warning]LLM 服务未初始化，跳过对话。[/warning]")
            return

        now = datetime.now()
        self._last_user_input_time = now
        self._user_input_times.append(now)

        if self._chat_history is None:
            # 首次对话：初始化上下文
            self._chat_start_time = now
            self._last_assistant_response_time = None
            self._chat_history = self.llm_service.build_chat_context(user_text)
        else:
            # 后续对话：追加用户消息到已有上下文
            self._chat_history.append(build_message(role="user", content=user_text))

        await self._run_llm_loop(self._chat_history)

    async def _run_llm_loop(self, chat_history: list):
        """
        LLM 循环架构核心。

        LLM 持续运行，每步可能输出文本（内心思考）和/或调用工具：
        - say(text): 对用户说话
        - wait(seconds): 暂停等待用户输入，超时或收到输入后继续
        - stop(): 结束循环，进入待机，直到用户下次输入
        - 不调用工具: 继续下一轮思考/生成

        每轮流程：
        1. 上下文管理：达到上限时自动压缩
        2. 情商 + Timing + 了解模块（并行）：分析用户情绪、对话时间节奏、检索用户特征
           *注：如果上次没有调用工具，跳过模块分析
        3. 调用主 LLM：基于完整上下文生成响应
        """
        consecutive_errors = 0
        last_had_tool_calls = True  # 第一次循环总是执行模块分析

        while True:
            # ── 上下文管理 ──
            await self._manage_context_length(chat_history)

            # ── 情商模块 + Timing 模块 + 了解模块（并行） ──
            # 只有上次调用了工具才重新分析（首次循环除外）
            if last_had_tool_calls:
                timing_info = build_timing_info(
                    self._chat_start_time,
                    self._last_user_input_time,
                    self._last_assistant_response_time,
                    self._user_input_times,
                )

                # 根据配置决定要执行的模块
                tasks = []
                status_text_parts = []

                if ENABLE_EMOTION_MODULE:
                    tasks.append(("eq", self.llm_service.analyze_emotion(chat_history)))
                    status_text_parts.append("🎭")
                if ENABLE_COGNITION_MODULE:
                    tasks.append(("cognition", self.llm_service.analyze_cognition(chat_history)))
                    status_text_parts.append("🧩")
                if ENABLE_TIMING_MODULE:
                    tasks.append(("timing", self.llm_service.analyze_timing(chat_history, timing_info)))
                    status_text_parts.append("⏱️🪞")
                if ENABLE_KNOWLEDGE_MODULE:
                    tasks.append(("knowledge", retrieve_relevant_knowledge(self.llm_service, chat_history)))
                    status_text_parts.append("👤")

                with console.status(
                    f"[info]{' '.join(status_text_parts)} {' + '.join(status_text_parts)} 模块并行分析中...[/info]",
                    spinner="dots",
                ):
                    results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

                # 解析结果
                eq_result, cognition_result, timing_result, knowledge_result = None, None, None, None
                result_idx = 0
                if ENABLE_EMOTION_MODULE:
                    eq_result = results[result_idx]
                    result_idx += 1
                if ENABLE_COGNITION_MODULE:
                    cognition_result = results[result_idx]
                    result_idx += 1
                if ENABLE_TIMING_MODULE:
                    timing_result = results[result_idx]
                    result_idx += 1
                if ENABLE_KNOWLEDGE_MODULE:
                    knowledge_result = results[result_idx]
                    result_idx += 1

                # 处理情商模块结果
                eq_analysis = ""
                if ENABLE_EMOTION_MODULE:
                    if isinstance(eq_result, Exception):
                        console.print(f"[warning]情商模块分析失败: {eq_result}[/warning]")
                    elif eq_result:
                        eq_analysis = eq_result
                        console.print(
                            Panel(
                                Markdown(eq_analysis),
                                title="🎭 情绪感知",
                                border_style="bright_yellow",
                                padding=(0, 1),
                                style="dim",
                            )
                        )

                # 处理认知模块结果
                cognition_analysis = ""
                if ENABLE_COGNITION_MODULE:
                    if isinstance(cognition_result, Exception):
                        console.print(f"[warning]认知模块分析失败: {cognition_result}[/warning]")
                    elif cognition_result:
                        cognition_analysis = cognition_result
                        console.print(
                            Panel(
                                Markdown(cognition_analysis),
                                title="🧩 意图感知",
                                border_style="bright_cyan",
                                padding=(0, 1),
                                style="dim",
                            )
                        )

                # 处理 Timing 模块结果（含自我反思功能）
                timing_analysis = ""
                if ENABLE_TIMING_MODULE:
                    if isinstance(timing_result, Exception):
                        console.print(f"[warning]Timing 模块分析失败: {timing_result}[/warning]")
                    elif timing_result:
                        timing_analysis = timing_result
                        console.print(
                            Panel(
                                Markdown(timing_analysis),
                                title="⏱️🪞 时间感知 & 自我反思",
                                border_style="bright_blue",
                                padding=(0, 1),
                                style="dim",
                            )
                        )

                # 处理了解模块结果
                knowledge_analysis = ""
                if ENABLE_KNOWLEDGE_MODULE:
                    if isinstance(knowledge_result, Exception):
                        console.print(f"[warning]了解模块分析失败: {knowledge_result}[/warning]")
                    elif knowledge_result:
                        knowledge_analysis = knowledge_result
                        console.print(
                            Panel(
                                Markdown(knowledge_analysis),
                                title="👤 用户特征",
                                border_style="bright_magenta",
                                padding=(0, 1),
                                style="dim",
                            )
                        )

                # 注入感知信息（作为 assistant 的感知消息）
                # 移除上一条感知消息（如果存在）
                remove_last_perception(chat_history)

                # 构建感知内容
                perception_parts = []
                if eq_analysis:
                    perception_parts.append(f"情绪感知\n{eq_analysis}")
                if cognition_analysis:
                    perception_parts.append(f"意图感知\n{cognition_analysis}")
                if timing_analysis:
                    perception_parts.append(f"时间感知 & 自我反思\n{timing_analysis}")
                if knowledge_analysis:
                    perception_parts.append(f"用户特征\n{knowledge_analysis}")

                if perception_parts:
                    # 添加感知消息（AI 的感知能力结果）
                    chat_history.append(
                        build_message(
                            role="assistant",
                            content="\n\n".join(perception_parts),
                            msg_type="perception",
                        )
                    )
            else:
                # 上次没有调用工具，跳过模块分析
                console.print("[muted]ℹ️  上次未调用工具，跳过模块分析[/muted]")

            # ── 调用 LLM ──
            with console.status("[info]💬 AI 正在思考...[/info]", spinner="dots"):
                try:
                    response = await self.llm_service.chat_loop_step(chat_history)
                    consecutive_errors = 0
                except Exception as e:
                    consecutive_errors += 1
                    console.print(f"[error]LLM 调用出错: {e}[/error]")
                    if consecutive_errors >= 3:
                        console.print("[error]连续出错，退出对话[/error]\n")
                        break
                    continue

            # 将 assistant 消息追加到历史
            chat_history.append(response.raw_message)
            self._last_assistant_response_time = datetime.now()


            # 显示内心思考（content 部分，淡色呈现）
            if response.content:
                console.print(
                    Panel(
                        Markdown(response.content),
                        title="💭 内心思考",
                        border_style="dim",
                        padding=(1, 2),
                        style="dim",
                    )
                )

            # ── 处理工具调用 ──
            if response.content and not response.tool_calls:
                last_had_tool_calls = False
                continue

            if response.tool_calls:
                should_stop = False
                ctx = self._build_tool_context()

                for tc in response.tool_calls:
                    if tc.name == "stop":
                        await handle_stop(tc, chat_history)
                        should_stop = True

                    elif tc.name == "reply":
                        reply = await self._generate_visible_reply(chat_history, response.content)
                        chat_history.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": "Visible reply generated and recorded.",
                            }
                        )
                        chat_history.append(
                            build_message(
                                role="user",
                                content=f"\u3010\u9ea6\u9ea6\u7684\u53d1\u8a00\u3011{reply}",
                            )
                        )

                    elif tc.name == "no_reply":
                        console.print("[muted]No visible reply this round.[/muted]")
                        chat_history.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": "No visible reply was sent for this round.",
                            }
                        )

                    elif tc.name == "wait":
                        tool_result = await handle_wait(tc, chat_history, ctx)
                        # 同步回 timing 时间戳
                        if ctx.last_user_input_time != self._last_user_input_time:
                            self._last_user_input_time = ctx.last_user_input_time
                        if tool_result.startswith("[[QUIT]]"):
                            should_stop = True

                    elif tc.name == "write_file":
                        await handle_write_file(tc, chat_history)

                    elif tc.name == "read_file":
                        await handle_read_file(tc, chat_history)

                    elif tc.name == "list_files":
                        await handle_list_files(tc, chat_history)

                    elif self._mcp_manager and self._mcp_manager.is_mcp_tool(tc.name):
                        await handle_mcp_tool(tc, chat_history, self._mcp_manager)

                    else:
                        await handle_unknown_tool(tc, chat_history)

                if should_stop:
                    console.print("[muted]对话暂停，等待新输入...[/muted]\n")
                    break

                # 调用了工具，下次循环需要重新分析模块
                last_had_tool_calls = True
            else:
                # LLM 未调用任何工具 → 继续下一轮思考
                # （不做任何额外操作，直接回到循环顶部再次调用 LLM）
                # 标记上次没有调用工具，下次循环跳过模块分析
                last_had_tool_calls = False
                continue

    # ──────── 主循环 ────────

    async def _init_mcp(self):
        """初始化 MCP 服务器连接，发现并注册外部工具。"""
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "mcp_config.json",
        )
        self._mcp_manager = await MCPManager.from_config(config_path)

        if self._mcp_manager and self.llm_service:
            mcp_tools = self._mcp_manager.get_openai_tools()
            if mcp_tools:
                self.llm_service.set_extra_tools(mcp_tools)
                summary = self._mcp_manager.get_tool_summary()
                console.print(
                    Panel(
                        f"已加载 {len(mcp_tools)} 个 MCP 工具:\n{summary}",
                        title="🔌 MCP 工具",
                        border_style="green",
                        padding=(0, 1),
                    )
                )

    async def _generate_visible_reply(self, chat_history: list, latest_thought: str) -> str:
        """Generate and emit a visible reply based on the latest thought."""
        if not self.llm_service or not latest_thought:
            return ""

        with console.status("[info]Generating visible reply...[/info]", spinner="dots"):
            reply = await self.llm_service.generate_reply(latest_thought, chat_history)

        console.print(
            Panel(
                Markdown(reply),
                title="MaiSaka",
                border_style="magenta",
                padding=(1, 2),
            )
        )
        return reply

    async def run(self):
        """主循环：直接输入文本即可对话"""
        # 根据配置决定是否初始化 MCP 服务器
        if ENABLE_MCP:
            await self._init_mcp()
        else:
            console.print("[muted]🔌 MCP 已禁用 (ENABLE_MCP=false)[/muted]")

        # 启动异步输入读取器
        self._reader.start(asyncio.get_event_loop())

        self._show_banner()

        try:
            while True:
                console.print("[bold cyan]> [/bold cyan]", end="")
                raw_input = await self._reader.get_line()

                if raw_input is None:  # EOF
                    console.print("\n[muted]再见！[/muted]")
                    break

                raw_input = raw_input.strip()
                if not raw_input:
                    continue

                await self._start_chat(raw_input)
        finally:
            if self._mcp_manager:
                await self._mcp_manager.close()



