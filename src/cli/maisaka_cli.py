"""
MaiSaka CLI and conversation loop.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import asyncio
import os
import time

from rich import box
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from src.know_u.knowledge import KnowledgeLearner, retrieve_relevant_knowledge
from src.know_u.knowledge_store import get_knowledge_store
from src.chat.message_receive.message import SessionMessage
from src.chat.replyer.maisaka_generator import MaisakaReplyGenerator
from src.config.config import config_manager, global_config
from src.mcp_module import MCPManager

from src.maisaka.chat_loop_service import MaisakaChatLoopService
from src.maisaka.message_adapter import build_message, format_speaker_content, remove_last_perception
from src.maisaka.tool_handlers import (
    ToolHandlerContext,
    handle_mcp_tool,
    handle_stop,
    handle_unknown_tool,
    handle_wait,
)

from .console import console
from .input_reader import InputReader


class BufferCLI:
    """Maisaka 命令行交互入口。"""

    def __init__(self) -> None:
        self._chat_loop_service: Optional[MaisakaChatLoopService] = None
        self._reply_generator = MaisakaReplyGenerator()
        self._reader = InputReader()
        self._chat_history: Optional[list[SessionMessage]] = None
        self._knowledge_store = get_knowledge_store()
        self._knowledge_learner = KnowledgeLearner("maisaka_cli")
        self._knowledge_min_messages_for_extraction = 10
        self._knowledge_min_extraction_interval = 30
        self._last_knowledge_extraction_time = 0.0

        knowledge_stats = self._knowledge_store.get_stats()
        if knowledge_stats["total_items"] > 0:
            console.print(f"[success][OK] Knowledge store: {knowledge_stats['total_items']} item(s)[/success]")
        else:
            console.print("[muted][OK] Knowledge store: initialized with no data[/muted]")

        self._chat_start_time: Optional[datetime] = None
        self._last_user_input_time: Optional[datetime] = None
        self._last_assistant_response_time: Optional[datetime] = None
        self._user_input_times: list[datetime] = []
        self._mcp_manager: Optional[MCPManager] = None
        self._init_llm()

    def _init_llm(self) -> None:
        """初始化 Maisaka 使用的聊天服务。"""
        thinking_env = os.getenv("ENABLE_THINKING", "").strip().lower()
        enable_thinking: Optional[bool] = True if thinking_env == "true" else False if thinking_env == "false" else None

        _ = enable_thinking
        self._chat_loop_service = MaisakaChatLoopService()

        model_name = self._get_current_model_name()
        console.print(f"[success][OK] LLM service initialized[/success] [muted](model: {model_name})[/muted]")

    @staticmethod
    def _get_current_model_name() -> str:
        """读取当前 planner 模型名。"""
        try:
            model_task_config = config_manager.get_model_config().model_task_config
            if model_task_config.planner.model_list:
                return model_task_config.planner.model_list[0]
        except Exception:
            pass
        return "unconfigured"

    def _build_tool_context(self) -> ToolHandlerContext:
        """构建工具处理的共享上下文。"""
        tool_context = ToolHandlerContext(
            reader=self._reader,
            user_input_times=self._user_input_times,
        )
        tool_context.last_user_input_time = self._last_user_input_time
        return tool_context

    def _show_banner(self) -> None:
        """渲染启动横幅。"""
        banner = Text()
        banner.append("MaiSaka", style="bold cyan")
        banner.append(" v2.0\n", style="muted")
        banner.append("Type to chat | Ctrl+C to exit", style="muted")

        console.print(Panel(banner, box=box.DOUBLE_EDGE, border_style="cyan", padding=(1, 2)))
        console.print()

    async def _start_chat(self, user_text: str) -> None:
        """追加用户输入并继续内部循环。"""
        if self._chat_loop_service is None:
            console.print("[warning]LLM service is not initialized; skipping chat.[/warning]")
            return

        now = datetime.now()
        self._last_user_input_time = now
        self._user_input_times.append(now)

        if self._chat_history is None:
            self._chat_start_time = now
            self._last_assistant_response_time = None
            self._chat_history = self._chat_loop_service.build_chat_context(user_text)
            self._trigger_knowledge_learning([self._chat_history[-1]])
        else:
            self._chat_history.append(
                build_message(
                    role="user",
                    content=format_speaker_content(
                        global_config.maisaka.user_name.strip() or "User",
                        user_text,
                        now,
                    ),
                )
            )
            self._trigger_knowledge_learning([self._chat_history[-1]])

        await self._run_llm_loop(self._chat_history)

    def _trigger_knowledge_learning(self, messages: list[SessionMessage]) -> None:
        """在 CLI 会话中按批次触发 knowledge 学习。"""
        if not global_config.maisaka.enable_knowledge_module:
            return

        self._knowledge_learner.add_messages(messages)

        elapsed = time.monotonic() - self._last_knowledge_extraction_time
        if elapsed < self._knowledge_min_extraction_interval:
            return

        cache_size = self._knowledge_learner.get_cache_size()
        if cache_size < self._knowledge_min_messages_for_extraction:
            return

        self._last_knowledge_extraction_time = time.monotonic()
        asyncio.create_task(self._run_knowledge_learning())

    async def _run_knowledge_learning(self) -> None:
        """后台执行 knowledge 学习，避免阻塞主对话。"""
        try:
            added_count = await self._knowledge_learner.learn()
            if added_count > 0 and global_config.maisaka.show_thinking:
                console.print(f"[muted]Knowledge learning added {added_count} item(s).[/muted]")
        except Exception as exc:
            console.print(f"[warning]Knowledge learning failed: {exc}[/warning]")

    async def _run_llm_loop(self, chat_history: list[SessionMessage]) -> None:
        """
        Main inner loop for the Maisaka planner.

        Each round may produce internal thoughts and optionally call tools:
        - reply(msg_id): generate a visible reply for the current round
        - no_reply(): skip visible output and continue the loop
        - wait(seconds): wait for new user input
        - stop(): stop the current inner loop and return to idle
        """
        if self._chat_loop_service is None:
            return

        consecutive_errors = 0
        last_had_tool_calls = True

        while True:
            if last_had_tool_calls:
                tasks = []
                status_text_parts = []

                if global_config.maisaka.enable_knowledge_module:
                    tasks.append(("knowledge", retrieve_relevant_knowledge(self._chat_loop_service, chat_history)))
                    status_text_parts.append("knowledge")

                with console.status(
                    f"[info]{' + '.join(status_text_parts)} analyzing...[/info]",
                    spinner="dots",
                ):
                    results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

                knowledge_analysis = ""
                if global_config.maisaka.enable_knowledge_module:
                    knowledge_result = results[0] if results else None
                    if isinstance(knowledge_result, Exception):
                        console.print(f"[warning]Knowledge analysis failed: {knowledge_result}[/warning]")
                    elif knowledge_result:
                        knowledge_analysis = knowledge_result
                        if global_config.maisaka.show_thinking:
                            console.print(
                                Panel(
                                    Markdown(knowledge_analysis),
                                    title="Knowledge",
                                    border_style="bright_magenta",
                                    padding=(0, 1),
                                    style="dim",
                                )
                            )

                remove_last_perception(chat_history)

                perception_parts = []
                if knowledge_analysis:
                    perception_parts.append(f"Knowledge\n{knowledge_analysis}")

                if perception_parts:
                    chat_history.append(
                        build_message(
                            role="assistant",
                            content="\n\n".join(perception_parts),
                            message_kind="perception",
                            source="assistant",
                        )
                    )
            elif global_config.maisaka.show_thinking:
                console.print("[muted]Skipping module analysis because the last round used no tools.[/muted]")

            with console.status("[info]AI is thinking...[/info]", spinner="dots"):
                try:
                    response = await self._chat_loop_service.chat_loop_step(chat_history)
                    consecutive_errors = 0
                except Exception as exc:
                    consecutive_errors += 1
                    console.print(f"[error]LLM call failed: {exc}[/error]")
                    if consecutive_errors >= 3:
                        console.print("[error]Too many consecutive errors. Exiting chat.[/error]\n")
                        break
                    continue

            chat_history.append(response.raw_message)
            self._last_assistant_response_time = datetime.now()

            if global_config.maisaka.show_thinking and response.content:
                console.print(
                    Panel(
                        Markdown(response.content),
                        title="Thought",
                        border_style="dim",
                        padding=(1, 2),
                        style="dim",
                    )
                )

            if response.content and not response.tool_calls:
                last_had_tool_calls = False
                continue

            if not response.tool_calls:
                last_had_tool_calls = False
                continue

            should_stop = False
            tool_context = self._build_tool_context()

            for tool_call in response.tool_calls:
                if tool_call.func_name == "stop":
                    await handle_stop(tool_call, chat_history)
                    should_stop = True

                elif tool_call.func_name == "reply":
                    reply = await self._generate_visible_reply(chat_history, response.content)
                    chat_history.append(
                        build_message(
                            role="tool",
                            content="Visible reply generated and recorded.",
                            source="tool",
                            tool_call_id=tool_call.call_id,
                        )
                    )
                    chat_history.append(
                        build_message(
                            role="user",
                            content=format_speaker_content(
                                global_config.bot.nickname.strip() or "MaiSaka",
                                reply,
                                datetime.now(),
                            ),
                            source="guided_reply",
                        )
                    )

                elif tool_call.func_name == "no_reply":
                    if global_config.maisaka.show_thinking:
                        console.print("[muted]No visible reply this round.[/muted]")
                    chat_history.append(
                        build_message(
                            role="tool",
                            content="No visible reply was sent for this round.",
                            source="tool",
                            tool_call_id=tool_call.call_id,
                        )
                    )

                elif tool_call.func_name == "wait":
                    tool_result = await handle_wait(tool_call, chat_history, tool_context)
                    if tool_context.last_user_input_time != self._last_user_input_time:
                        self._last_user_input_time = tool_context.last_user_input_time
                    if tool_result.startswith("[[QUIT]]"):
                        should_stop = True

                elif self._mcp_manager and self._mcp_manager.is_mcp_tool(tool_call.func_name):
                    await handle_mcp_tool(tool_call, chat_history, self._mcp_manager)

                else:
                    await handle_unknown_tool(tool_call, chat_history)

            if should_stop:
                console.print("[muted]Conversation paused. Waiting for new input...[/muted]\n")
                break

            last_had_tool_calls = True

    async def _init_mcp(self) -> None:
        """初始化 MCP 服务并注册暴露的工具。"""
        config_path = Path(__file__).resolve().parents[2] / "config" / "mcp_config.json"
        self._mcp_manager = await MCPManager.from_config(str(config_path))

        if self._mcp_manager and self._chat_loop_service:
            mcp_tools = self._mcp_manager.get_openai_tools()
            if mcp_tools:
                self._chat_loop_service.set_extra_tools(mcp_tools)
                summary = self._mcp_manager.get_tool_summary()
                console.print(
                    Panel(
                        f"Loaded {len(mcp_tools)} MCP tool(s):\n{summary}",
                        title="MCP Tools",
                        border_style="green",
                        padding=(0, 1),
                    )
                )

    async def _generate_visible_reply(self, chat_history: list[SessionMessage], latest_thought: str) -> str:
        """根据最新思考生成并输出可见回复。"""
        if not latest_thought:
            return ""

        with console.status("[info]Generating visible reply...[/info]", spinner="dots"):
            success, result = await self._reply_generator.generate_reply_with_context(
                reply_reason=latest_thought,
                chat_history=chat_history,
            )
            if success and result.text_fragments:
                reply = result.text_fragments[0]
            else:
                reply = "..."

        console.print(
            Panel(
                Markdown(reply),
                title="MaiSaka",
                border_style="magenta",
                padding=(1, 2),
            )
        )

        return reply

    async def run(self) -> None:
        """主交互循环。"""
        if global_config.maisaka.enable_mcp:
            await self._init_mcp()
        else:
            console.print("[muted]MCP is disabled (ENABLE_MCP=false)[/muted]")

        self._reader.start(asyncio.get_event_loop())
        self._show_banner()

        try:
            while True:
                console.print("[bold cyan]> [/bold cyan]", end="")
                raw_input = await self._reader.get_line()

                if raw_input is None:
                    console.print("\n[muted]Goodbye![/muted]")
                    break

                raw_input = raw_input.strip()
                if not raw_input:
                    continue

                await self._start_chat(raw_input)
        finally:
            if self._mcp_manager:
                await self._mcp_manager.close()
