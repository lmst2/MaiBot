"""
MaiSaka tool handlers.
"""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import json as _json
import os

from rich.markdown import Markdown
from rich.panel import Panel

from .config import console
from .input_reader import InputReader
from .llm_service import MaiSakaLLMService
from .replyer import Replyer

if TYPE_CHECKING:
    from .mcp_client import MCPManager


MAI_FILES_DIR = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mai_files"))

_replyer: Optional[Replyer] = None


def get_replyer(llm_service: MaiSakaLLMService) -> Replyer:
    """Return a shared replyer instance."""
    global _replyer
    if _replyer is None:
        _replyer = Replyer(llm_service)
    elif _replyer._llm_service is None:
        _replyer.set_llm_service(llm_service)
    return _replyer


class ToolHandlerContext:
    """Shared context for tool handlers."""

    def __init__(
        self,
        llm_service: MaiSakaLLMService,
        reader: InputReader,
        user_input_times: list[datetime],
    ) -> None:
        self.llm_service = llm_service
        self.reader = reader
        self.user_input_times = user_input_times
        self.last_user_input_time: Optional[datetime] = None


async def handle_send_message(tc: Any, chat_history: list[dict[str, Any]], ctx: ToolHandlerContext) -> None:
    """Backward-compatible handler for legacy send-message style tools."""
    reason = tc.arguments.get("reason", "")
    console.print("[accent]Calling tool: send_message(...)[/accent]")

    if not reason:
        chat_history.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": "Missing required argument: reason",
            }
        )
        return

    console.print(
        Panel(
            Markdown(reason),
            title="Reply Reason",
            border_style="dim",
            padding=(0, 1),
            style="dim",
        )
    )

    with console.status("[info]Generating visible reply...[/info]", spinner="dots"):
        replyer = get_replyer(ctx.llm_service)
        reply = await replyer.reply(reason, chat_history)

    console.print(
        Panel(
            Markdown(reply),
            title="MaiSaka",
            border_style="magenta",
            padding=(1, 2),
        )
    )
    chat_history.append(
        {
            "role": "tool",
            "tool_call_id": tc.id,
            "content": f"Visible reply generated:\n{reply}",
        }
    )


async def handle_stop(tc: Any, chat_history: list[dict[str, Any]]) -> None:
    """Handle the stop tool."""
    console.print("[accent]Calling tool: stop()[/accent]")
    chat_history.append(
        {
            "role": "tool",
            "tool_call_id": tc.id,
            "content": "Conversation loop will stop after this round.",
        }
    )


async def handle_wait(tc: Any, chat_history: list[dict[str, Any]], ctx: ToolHandlerContext) -> str:
    """Handle the wait tool."""
    seconds = tc.arguments.get("seconds", 30)
    seconds = max(5, min(seconds, 300))
    console.print(f"[accent]Calling tool: wait({seconds})[/accent]")

    tool_result = await _do_wait(seconds, ctx)
    chat_history.append(
        {
            "role": "tool",
            "tool_call_id": tc.id,
            "content": tool_result,
        }
    )
    return tool_result


async def _do_wait(seconds: int, ctx: ToolHandlerContext) -> str:
    """Wait for user input with a timeout."""
    console.print(f"[muted]Waiting for user input (timeout: {seconds}s)...[/muted]")
    console.print("[bold magenta]> [/bold magenta]", end="")

    user_input = await ctx.reader.get_line(timeout=seconds)

    if user_input is None:
        console.print()
        console.print("[muted]Wait timeout[/muted]")
        return "Wait timed out; no user input received."

    user_input = user_input.strip()
    if not user_input:
        return "User submitted an empty input."

    now = datetime.now()
    ctx.last_user_input_time = now
    ctx.user_input_times.append(now)

    if user_input.lower() in ("/quit", "/exit", "/q"):
        return "[[QUIT]] User requested to exit."

    return f"User input received: {user_input}"


async def handle_mcp_tool(tc: Any, chat_history: list[dict[str, Any]], mcp_manager: "MCPManager") -> None:
    """Handle an MCP tool call."""
    args_str = _json.dumps(tc.arguments, ensure_ascii=False)
    args_preview = args_str if len(args_str) <= 120 else args_str[:120] + "..."
    console.print(f"[accent]Calling MCP tool: {tc.name}({args_preview})[/accent]")

    with console.status(f"[info]Running MCP tool {tc.name}...[/info]", spinner="dots"):
        result = await mcp_manager.call_tool(tc.name, tc.arguments)

    display_text = result if len(result) <= 800 else result[:800] + "\n... (truncated)"
    console.print(
        Panel(
            display_text,
            title=f"MCP: {tc.name}",
            border_style="bright_green",
            padding=(0, 1),
        )
    )
    chat_history.append(
        {
            "role": "tool",
            "tool_call_id": tc.id,
            "content": result,
        }
    )


async def handle_unknown_tool(tc: Any, chat_history: list[dict[str, Any]]) -> None:
    """Handle an unknown tool call."""
    console.print(f"[accent]Calling unknown tool: {tc.name}({tc.arguments})[/accent]")
    chat_history.append(
        {
            "role": "tool",
            "tool_call_id": tc.id,
            "content": f"Unknown tool: {tc.name}",
        }
    )


async def handle_write_file(tc: Any, chat_history: list[dict[str, Any]]) -> None:
    """Write a file under the local mai_files workspace."""
    filename = tc.arguments.get("filename", "")
    content = tc.arguments.get("content", "")
    console.print(f'[accent]Calling tool: write_file("{filename}")[/accent]')

    MAI_FILES_DIR.mkdir(parents=True, exist_ok=True)
    file_path = MAI_FILES_DIR / filename

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)

        file_size = file_path.stat().st_size
        console.print(
            Panel(
                f"Path: {filename}\nSize: {file_size} bytes",
                title="File Written",
                border_style="green",
                padding=(0, 1),
            )
        )
        chat_history.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": f"File written successfully: {filename} ({file_size} bytes)",
            }
        )
    except Exception as exc:
        error_msg = f"Failed to write file: {exc}"
        console.print(f"[error]{error_msg}[/error]")
        chat_history.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": error_msg,
            }
        )


async def handle_read_file(tc: Any, chat_history: list[dict[str, Any]]) -> None:
    """Read a file from the local mai_files workspace."""
    filename = tc.arguments.get("filename", "")
    console.print(f'[accent]Calling tool: read_file("{filename}")[/accent]')

    file_path = MAI_FILES_DIR / filename

    try:
        if not file_path.exists():
            error_msg = f"File does not exist: {filename}"
            console.print(f"[warning]{error_msg}[/warning]")
            chat_history.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": error_msg,
                }
            )
            return

        if not file_path.is_file():
            error_msg = f"Path is not a file: {filename}"
            console.print(f"[warning]{error_msg}[/warning]")
            chat_history.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": error_msg,
                }
            )
            return

        with open(file_path, "r", encoding="utf-8") as file:
            file_content = file.read()

        display_content = file_content if len(file_content) <= 1000 else file_content[:1000] + "\n... (truncated)"
        console.print(
            Panel(
                display_content,
                title=f"Read File: {filename}",
                border_style="blue",
                padding=(0, 1),
            )
        )
        chat_history.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": f"File content of {filename}:\n{file_content}",
            }
        )
    except Exception as exc:
        error_msg = f"Failed to read file: {exc}"
        console.print(f"[error]{error_msg}[/error]")
        chat_history.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": error_msg,
            }
        )


async def handle_list_files(tc: Any, chat_history: list[dict[str, Any]]) -> None:
    """List files under the local mai_files workspace."""
    console.print("[accent]Calling tool: list_files()[/accent]")

    try:
        MAI_FILES_DIR.mkdir(parents=True, exist_ok=True)

        files_info: list[dict[str, Any]] = []
        for item in MAI_FILES_DIR.rglob("*"):
            if item.is_file():
                stat = item.stat()
                files_info.append(
                    {
                        "name": str(item.relative_to(MAI_FILES_DIR)),
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

        if not files_info:
            result_text = "No files found under mai_files."
        else:
            files_info.sort(key=lambda item: item["name"])
            lines = [f"Found {len(files_info)} file(s):\n"]
            for item in files_info:
                lines.append(f"- {item['name']} ({item['size']} bytes, modified {item['modified']})")
            result_text = "\n".join(lines)

        console.print(
            Panel(
                result_text,
                title="File List",
                border_style="cyan",
                padding=(0, 1),
            )
        )
        chat_history.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_text,
            }
        )
    except Exception as exc:
        error_msg = f"Failed to list files: {exc}"
        console.print(f"[error]{error_msg}[/error]")
        chat_history.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": error_msg,
            }
        )


try:
    MAI_FILES_DIR.mkdir(parents=True, exist_ok=True)
except Exception as exc:
    console.print(f"[warning]Failed to initialize mai_files directory: {exc}[/warning]")
