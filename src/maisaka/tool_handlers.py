"""
MaiSaka 工具处理器。
"""

from datetime import datetime
from typing import TYPE_CHECKING, Optional

import json as _json

from rich.panel import Panel

from src.cli.console import console
from src.cli.input_reader import InputReader
from src.llm_models.payload_content.tool_option import ToolCall

from .context_messages import LLMContextMessage, ToolResultMessage

if TYPE_CHECKING:
    from src.mcp_module import MCPManager


class ToolHandlerContext:
    """工具处理器共享上下文。"""

    def __init__(
        self,
        reader: InputReader,
        user_input_times: list[datetime],
    ) -> None:
        self.reader = reader
        self.user_input_times = user_input_times
        self.last_user_input_time: Optional[datetime] = None


async def handle_stop(tc: ToolCall, chat_history: list[LLMContextMessage]) -> None:
    """处理 stop 工具。"""
    console.print("[accent]调用工具: stop()[/accent]")
    chat_history.append(
        ToolResultMessage(
            content="当前轮次结束后将停止对话循环。",
            timestamp=datetime.now(),
            tool_call_id=tc.call_id,
            tool_name=tc.func_name,
        )
    )


async def handle_wait(tc: ToolCall, chat_history: list[LLMContextMessage], ctx: ToolHandlerContext) -> str:
    """处理 wait 工具。"""
    seconds = (tc.args or {}).get("seconds", 30)
    seconds = max(5, min(seconds, 300))
    console.print(f"[accent]调用工具: wait({seconds})[/accent]")

    tool_result = await _do_wait(seconds, ctx)
    chat_history.append(
        ToolResultMessage(
            content=tool_result,
            timestamp=datetime.now(),
            tool_call_id=tc.call_id,
            tool_name=tc.func_name,
        )
    )
    return tool_result


async def _do_wait(seconds: int, ctx: ToolHandlerContext) -> str:
    """等待用户输入，支持超时。"""
    console.print(f"[muted]等待用户输入中（超时: {seconds} 秒）...[/muted]")
    console.print("[bold magenta]> [/bold magenta]", end="")

    user_input = await ctx.reader.get_line(timeout=seconds)

    if user_input is None:
        console.print()
        console.print("[muted]等待超时[/muted]")
        return "等待超时，未收到用户输入。"

    user_input = user_input.strip()
    if not user_input:
        return "用户提交了空输入。"

    now = datetime.now()
    ctx.last_user_input_time = now
    ctx.user_input_times.append(now)

    if user_input.lower() in ("/quit", "/exit", "/q"):
        return "[[QUIT]] 用户请求退出。"

    return f"已收到用户输入: {user_input}"


async def handle_mcp_tool(tc: ToolCall, chat_history: list[LLMContextMessage], mcp_manager: "MCPManager") -> None:
    """处理 MCP 工具调用。"""
    args_str = _json.dumps(tc.args or {}, ensure_ascii=False)
    args_preview = args_str if len(args_str) <= 120 else args_str[:120] + "..."
    console.print(f"[accent]调用 MCP 工具: {tc.func_name}({args_preview})[/accent]")

    with console.status(f"[info]正在执行 MCP 工具 {tc.func_name}...[/info]", spinner="dots"):
        result = await mcp_manager.call_tool(tc.func_name, tc.args or {})

    display_text = result if len(result) <= 800 else result[:800] + "\n...（已截断）"
    console.print(
        Panel(
            display_text,
            title=f"MCP: {tc.func_name}",
            border_style="bright_green",
            padding=(0, 1),
        )
    )
    chat_history.append(
        ToolResultMessage(
            content=result,
            timestamp=datetime.now(),
            tool_call_id=tc.call_id,
            tool_name=tc.func_name,
        )
    )


async def handle_unknown_tool(tc: ToolCall, chat_history: list[LLMContextMessage]) -> None:
    """处理未知工具调用。"""
    console.print(f"[accent]调用未知工具: {tc.func_name}({tc.args})[/accent]")
    chat_history.append(
        ToolResultMessage(
            content=f"未知工具: {tc.func_name}",
            timestamp=datetime.now(),
            tool_call_id=tc.call_id,
            tool_name=tc.func_name,
        )
    )
