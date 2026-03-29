"""
MaiSaka - 单个 MCP 服务器连接管理
封装单个 MCP 服务器的连接生命周期：连接 → 发现工具 → 调用工具 → 断开。
"""

from contextlib import AsyncExitStack
from typing import Any, Optional

from src.cli.console import console

from .config import MCPServerConfig

# ──────────────────── MCP SDK 可选导入 ────────────────────
#
# mcp 是可选依赖。如果未安装，MCP_AVAILABLE = False，
# MCPManager.from_config() 会检测到并返回 None，不影响主程序运行。

try:
    from mcp import ClientSession

    try:
        from mcp.client.stdio import StdioServerParameters
    except ImportError:
        from mcp import StdioServerParameters  # type: ignore[attr-defined]

    from mcp.client.stdio import stdio_client

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    ClientSession = None  # type: ignore[assignment,misc]
    StdioServerParameters = None  # type: ignore[assignment,misc]
    stdio_client = None  # type: ignore[assignment]

try:
    from mcp.client.sse import sse_client

    SSE_AVAILABLE = True
except ImportError:
    SSE_AVAILABLE = False
    sse_client = None  # type: ignore[assignment]


class MCPConnection:
    """
    管理单个 MCP 服务器的连接生命周期。

    支持两种传输方式：
    - Stdio: 启动子进程，通过 stdin/stdout 通信
    - SSE: 连接远程 HTTP SSE 端点
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.session: Optional[Any] = None  # mcp.ClientSession
        self.tools: list = []  # mcp Tool objects
        self._exit_stack = AsyncExitStack()

    async def connect(self) -> bool:
        """
        连接到 MCP 服务器并发现可用工具。

        Returns:
            True 表示连接成功，False 表示失败。
        """
        if not MCP_AVAILABLE:
            console.print("[warning]⚠️ 未安装 mcp SDK，请运行: pip install mcp[/warning]")
            return False

        try:
            await self._exit_stack.__aenter__()

            if self.config.transport_type == "stdio":
                read_stream, write_stream = await self._connect_stdio()
            elif self.config.transport_type == "sse":
                read_stream, write_stream = await self._connect_sse()
            else:
                console.print(f"[warning]MCP '{self.config.name}': 未知传输类型[/warning]")
                return False

            # 创建并初始化 MCP 会话
            self.session = await self._exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
            await self.session.initialize()

            # 发现工具
            result = await self.session.list_tools()
            self.tools = result.tools if hasattr(result, "tools") else []

            return True

        except Exception as e:
            console.print(f"[warning]⚠️ MCP 服务器 '{self.config.name}' 连接失败: {e}[/warning]")
            await self.close()
            return False

    async def _connect_stdio(self):
        """建立 Stdio 传输连接。"""
        params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
            env=self.config.env,
        )
        return await self._exit_stack.enter_async_context(stdio_client(params))

    async def _connect_sse(self):
        """建立 SSE 传输连接。"""
        if not SSE_AVAILABLE:
            raise ImportError("SSE 传输需要额外依赖，请运行: pip install mcp[sse]")
        return await self._exit_stack.enter_async_context(sse_client(url=self.config.url, headers=self.config.headers))

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """
        调用 MCP 工具并返回结果文本。

        Args:
            tool_name:  工具名称
            arguments:  工具参数字典

        Returns:
            工具执行结果的文本表示。
        """
        if not self.session:
            return f"MCP 服务器 '{self.config.name}' 未连接"

        result = await self.session.call_tool(tool_name, arguments=arguments)

        # 将结果内容转换为文本
        parts: list[str] = []
        for content in result.content:
            if hasattr(content, "text"):
                parts.append(content.text)
            elif hasattr(content, "data"):
                # 二进制/图片内容，展示类型信息
                content_type = getattr(content, "mimeType", "unknown")
                parts.append(f"[{content_type} 二进制内容]")
            elif hasattr(content, "type"):
                parts.append(f"[{content.type} 内容]")

        return "\n".join(parts) if parts else "工具执行成功（无输出）"

    async def close(self):
        """关闭连接并释放资源。"""
        try:
            await self._exit_stack.aclose()
        except Exception:
            pass
        self.session = None
        self.tools = []
