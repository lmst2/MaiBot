"""
MaiSaka - MCP 管理器
管理所有 MCP 服务器连接，提供统一的工具发现与调用接口。
"""

from typing import Optional

from src.cli.console import console

from .config import DEFAULT_MCP_CONFIG_PATH, MCPServerConfig, load_mcp_config
from .connection import MCPConnection, MCP_AVAILABLE

# 内置工具名称集合 —— MCP 工具不允许与这些名称冲突
BUILTIN_TOOL_NAMES = frozenset(
    {
        "reply",
        "no_reply",
        "wait",
        "stop",
        "create_table",
        "list_tables",
        "view_table",
    }
)


class MCPManager:
    """
    MCP 服务器连接管理器。

    职责：
    - 根据配置文件连接所有 MCP 服务器
    - 将 MCP 工具转换为 OpenAI function calling 格式
    - 路由工具调用到正确的 MCP 服务器
    - 统一管理连接生命周期
    """

    def __init__(self):
        self._connections: dict[str, MCPConnection] = {}  # server_name → connection
        self._tool_to_server: dict[str, str] = {}  # tool_name → server_name

    # ──────── 工厂方法 ────────

    @classmethod
    async def from_config(
        cls,
        config_path: str = str(DEFAULT_MCP_CONFIG_PATH),
    ) -> Optional["MCPManager"]:
        """
        从配置文件创建并初始化 MCPManager。

        Args:
            config_path: mcp_config.json 文件路径

        Returns:
            初始化完成的 MCPManager；无配置或全部连接失败时返回 None。
        """
        configs = load_mcp_config(config_path)
        if not configs:
            return None

        if not MCP_AVAILABLE:
            console.print("[warning]⚠️ 发现 MCP 配置但未安装 mcp SDK，请运行: pip install mcp[/warning]")
            return None

        manager = cls()
        await manager._connect_all(configs)

        if not manager._connections:
            console.print("[warning]⚠️ 所有 MCP 服务器连接失败[/warning]")
            return None

        return manager

    # ──────── 连接管理 ────────

    async def _connect_all(self, configs: list[MCPServerConfig]):
        """连接所有配置的 MCP 服务器，跳过失败的连接。"""
        for cfg in configs:
            conn = MCPConnection(cfg)
            success = await conn.connect()
            if not success:
                continue

            self._connections[cfg.name] = conn

            # 注册工具，检查冲突
            registered = 0
            for tool in conn.tools:
                tool_name = tool.name

                if tool_name in BUILTIN_TOOL_NAMES:
                    console.print(
                        f"[warning]⚠️ MCP 工具 '{tool_name}' (来自 {cfg.name}) 与内置工具冲突，已跳过[/warning]"
                    )
                    continue

                if tool_name in self._tool_to_server:
                    existing_server = self._tool_to_server[tool_name]
                    console.print(
                        f"[warning]⚠️ MCP 工具 '{tool_name}' "
                        f"(来自 {cfg.name}) 与 {existing_server} 冲突，已跳过[/warning]"
                    )
                    continue

                self._tool_to_server[tool_name] = cfg.name
                registered += 1

            console.print(
                f"[success]✓ MCP 服务器 '{cfg.name}' 已连接[/success] [muted]({registered} 个工具已注册)[/muted]"
            )

    # ──────── 工具发现 ────────

    def get_openai_tools(self) -> list[dict]:
        """
        将所有已注册的 MCP 工具转换为 OpenAI function calling 格式。

        Returns:
            OpenAI tools 格式的工具定义列表。
        """
        tools: list[dict] = []

        for server_name, conn in self._connections.items():
            for tool in conn.tools:
                # 只包含成功注册的工具
                if tool.name not in self._tool_to_server:
                    continue
                if self._tool_to_server[tool.name] != server_name:
                    continue

                # MCP inputSchema → OpenAI parameters
                parameters = (
                    dict(tool.inputSchema)
                    if hasattr(tool, "inputSchema") and tool.inputSchema
                    else {"type": "object", "properties": {}}
                )
                # 移除 $schema 字段（部分 MCP 服务器会带上，OpenAI 不接受）
                parameters.pop("$schema", None)

                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": (tool.description or f"MCP tool from {server_name}"),
                            "parameters": parameters,
                        },
                    }
                )

        return tools

    # ──────── 工具调用 ────────

    def is_mcp_tool(self, tool_name: str) -> bool:
        """判断工具名是否为已注册的 MCP 工具。"""
        return tool_name in self._tool_to_server

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """
        调用指定的 MCP 工具。

        自动路由到正确的 MCP 服务器。

        Args:
            tool_name:  工具名称
            arguments:  工具参数

        Returns:
            工具执行结果文本。
        """
        server_name = self._tool_to_server.get(tool_name)
        if not server_name or server_name not in self._connections:
            return f"MCP 工具 '{tool_name}' 未找到"

        conn = self._connections[server_name]
        try:
            return await conn.call_tool(tool_name, arguments)
        except Exception as e:
            return f"MCP 工具 '{tool_name}' 执行失败: {e}"

    # ──────── 信息展示 ────────

    def get_tool_summary(self) -> str:
        """获取所有已注册 MCP 工具的摘要信息。"""
        parts: list[str] = []
        for server_name, conn in self._connections.items():
            tool_names = [
                t.name
                for t in conn.tools
                if t.name in self._tool_to_server and self._tool_to_server[t.name] == server_name
            ]
            if tool_names:
                parts.append(f"  • {server_name}: {', '.join(tool_names)}")
        return "\n".join(parts)

    @property
    def server_count(self) -> int:
        """已连接的 MCP 服务器数量。"""
        return len(self._connections)

    @property
    def tool_count(self) -> int:
        """已注册的 MCP 工具总数。"""
        return len(self._tool_to_server)

    # ──────── 生命周期 ────────

    async def close(self):
        """关闭所有 MCP 服务器连接。"""
        for conn in self._connections.values():
            await conn.close()
        self._connections.clear()
        self._tool_to_server.clear()
