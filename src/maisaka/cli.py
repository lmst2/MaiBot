"""
MaiSaka CLI and conversation loop.
"""

from datetime import datetime
from typing import Optional

import asyncio
import os

from rich import box
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from src.common.data_models.mai_message_data_model import MaiMessage
from src.config.config import global_config

from .config import (
    ENABLE_COGNITION_MODULE,
    ENABLE_EMOTION_MODULE,
    ENABLE_KNOWLEDGE_MODULE,
    ENABLE_MCP,
    SHOW_THINKING,
    USER_NAME,
    console,
)
from .input_reader import InputReader
from .knowledge import retrieve_relevant_knowledge
from .knowledge_store import get_knowledge_store
from .llm_service import MaiSakaLLMService, build_message, remove_last_perception
from .message_adapter import format_speaker_content
from .mcp_client import MCPManager
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
    """Command line interface for Maisaka."""

    def __init__(self):
        self.llm_service: Optional[MaiSakaLLMService] = None
        self._reader = InputReader()
        self._chat_history: Optional[list[MaiMessage]] = None
        self._knowledge_store = get_knowledge_store()

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

    def _init_llm(self):
        """Initialize the LLM service from the main project config."""
        thinking_env = os.getenv("ENABLE_THINKING", "").strip().lower()
        enable_thinking: Optional[bool] = True if thinking_env == "true" else False if thinking_env == "false" else None

        self.llm_service = MaiSakaLLMService(
            api_key="",
            base_url=None,
            model="",
            enable_thinking=enable_thinking,
        )

        model_name = self.llm_service._model_name
        console.print(f"[success][OK] LLM service initialized[/success] [muted](model: {model_name})[/muted]")

    def _build_tool_context(self) -> ToolHandlerContext:
        """Build the shared tool handler context."""
        ctx = ToolHandlerContext(
            llm_service=self.llm_service,
            reader=self._reader,
            user_input_times=self._user_input_times,
        )
        ctx.last_user_input_time = self._last_user_input_time
        return ctx

    def _show_banner(self):
        """Render the startup banner."""
        banner = Text()
        banner.append("MaiSaka", style="bold cyan")
        banner.append(" v2.0\n", style="muted")
        banner.append("Type to chat | Ctrl+C to exit", style="muted")

        console.print(Panel(banner, box=box.DOUBLE_EDGE, border_style="cyan", padding=(1, 2)))
        console.print()

    async def _start_chat(self, user_text: str):
        """Append user input and continue the inner loop."""
        if not self.llm_service:
            console.print("[warning]LLM service is not initialized; skipping chat.[/warning]")
            return

        now = datetime.now()
        self._last_user_input_time = now
        self._user_input_times.append(now)

        if self._chat_history is None:
            self._chat_start_time = now
            self._last_assistant_response_time = None
            self._chat_history = self.llm_service.build_chat_context(user_text)
        else:
            self._chat_history.append(
                build_message(
                    role="user",
                    content=format_speaker_content(USER_NAME, user_text, now),
                )
            )

        await self._run_llm_loop(self._chat_history)

    async def _run_llm_loop(self, chat_history: list[MaiMessage]):
        """
        Main inner loop for the Maisaka planner.

        Each round may produce internal thoughts and optionally call tools:
        - reply(): generate a visible reply for the current round
        - no_reply(): skip visible output and continue the loop
        - wait(seconds): wait for new user input
        - stop(): stop the current inner loop and return to idle

        Per round:
        1. Run enabled analysis modules in parallel when the previous round used tools.
        2. Call the planner model with the current history.
        3. Append the assistant thought and execute any requested tools.
        """
        consecutive_errors = 0
        last_had_tool_calls = True

        while True:
            if last_had_tool_calls:
                tasks = []
                status_text_parts = []

                if ENABLE_EMOTION_MODULE:
                    tasks.append(("eq", self.llm_service.analyze_emotion(chat_history)))
                    status_text_parts.append("emotion")
                if ENABLE_COGNITION_MODULE:
                    tasks.append(("cognition", self.llm_service.analyze_cognition(chat_history)))
                    status_text_parts.append("cognition")
                if ENABLE_KNOWLEDGE_MODULE:
                    tasks.append(("knowledge", retrieve_relevant_knowledge(self.llm_service, chat_history)))
                    status_text_parts.append("knowledge")

                with console.status(
                    f"[info]{' + '.join(status_text_parts)} analyzing...[/info]",
                    spinner="dots",
                ):
                    results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

                eq_result, cognition_result, knowledge_result = None, None, None
                result_idx = 0
                if ENABLE_EMOTION_MODULE:
                    eq_result = results[result_idx]
                    result_idx += 1
                if ENABLE_COGNITION_MODULE:
                    cognition_result = results[result_idx]
                    result_idx += 1
                if ENABLE_KNOWLEDGE_MODULE:
                    knowledge_result = results[result_idx]
                    result_idx += 1

                eq_analysis = ""
                if ENABLE_EMOTION_MODULE:
                    if isinstance(eq_result, Exception):
                        console.print(f"[warning]Emotion analysis failed: {eq_result}[/warning]")
                    elif eq_result:
                        eq_analysis = eq_result
                        if SHOW_THINKING:
                            console.print(
                                Panel(
                                    Markdown(eq_analysis),
                                    title="Emotion",
                                    border_style="bright_yellow",
                                    padding=(0, 1),
                                    style="dim",
                                )
                            )

                cognition_analysis = ""
                if ENABLE_COGNITION_MODULE:
                    if isinstance(cognition_result, Exception):
                        console.print(f"[warning]Cognition analysis failed: {cognition_result}[/warning]")
                    elif cognition_result:
                        cognition_analysis = cognition_result
                        if SHOW_THINKING:
                            console.print(
                                Panel(
                                    Markdown(cognition_analysis),
                                    title="Cognition",
                                    border_style="bright_cyan",
                                    padding=(0, 1),
                                    style="dim",
                                )
                            )

                knowledge_analysis = ""
                if ENABLE_KNOWLEDGE_MODULE:
                    if isinstance(knowledge_result, Exception):
                        console.print(f"[warning]Knowledge analysis failed: {knowledge_result}[/warning]")
                    elif knowledge_result:
                        knowledge_analysis = knowledge_result
                        if SHOW_THINKING:
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
                if eq_analysis:
                    perception_parts.append(f"Emotion\n{eq_analysis}")
                if cognition_analysis:
                    perception_parts.append(f"Cognition\n{cognition_analysis}")
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
            else:
                if SHOW_THINKING:
                    console.print("[muted]Skipping module analysis because the last round used no tools.[/muted]")

            with console.status("[info]AI is thinking...[/info]", spinner="dots"):
                try:
                    response = await self.llm_service.chat_loop_step(chat_history)
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

            if SHOW_THINKING and response.content:
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

            if response.tool_calls:
                should_stop = False
                ctx = self._build_tool_context()

                for tc in response.tool_calls:
                    if tc.func_name == "stop":
                        await handle_stop(tc, chat_history)
                        should_stop = True

                    elif tc.func_name == "reply":
                        reply = await self._generate_visible_reply(chat_history, response.content)
                        chat_history.append(
                            build_message(
                                role="tool",
                                content="Visible reply generated and recorded.",
                                source="tool",
                                tool_call_id=tc.call_id,
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

                    elif tc.func_name == "no_reply":
                        if SHOW_THINKING:
                            console.print("[muted]No visible reply this round.[/muted]")
                        chat_history.append(
                            build_message(
                                role="tool",
                                content="No visible reply was sent for this round.",
                                source="tool",
                                tool_call_id=tc.call_id,
                            )
                        )

                    elif tc.func_name == "wait":
                        tool_result = await handle_wait(tc, chat_history, ctx)
                        if ctx.last_user_input_time != self._last_user_input_time:
                            self._last_user_input_time = ctx.last_user_input_time
                        if tool_result.startswith("[[QUIT]]"):
                            should_stop = True

                    elif tc.func_name == "write_file":
                        await handle_write_file(tc, chat_history)

                    elif tc.func_name == "read_file":
                        await handle_read_file(tc, chat_history)

                    elif tc.func_name == "list_files":
                        await handle_list_files(tc, chat_history)

                    elif self._mcp_manager and self._mcp_manager.is_mcp_tool(tc.func_name):
                        await handle_mcp_tool(tc, chat_history, self._mcp_manager)

                    else:
                        await handle_unknown_tool(tc, chat_history)

                if should_stop:
                    console.print("[muted]Conversation paused. Waiting for new input...[/muted]\n")
                    break

                last_had_tool_calls = True
            else:
                last_had_tool_calls = False
                continue

    async def _init_mcp(self):
        """Initialize MCP servers and register exposed tools."""
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
                        f"Loaded {len(mcp_tools)} MCP tool(s):\n{summary}",
                        title="MCP Tools",
                        border_style="green",
                        padding=(0, 1),
                    )
                )

    async def _generate_visible_reply(self, chat_history: list[MaiMessage], latest_thought: str) -> str:
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
        """Main interactive loop."""
        if ENABLE_MCP:
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
