"""
MaiSaka - MCP 管理器
管理所有 MCP 服务器连接，提供统一的工具发现与调用接口。
"""

from typing import Any, Optional

from src.cli.console import console
from src.core.tooling import (
    ToolExecutionResult,
    ToolInvocation,
    ToolSpec,
    build_tool_detailed_description,
)

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

    async def _connect_all(self, configs: list[MCPServerConfig]) -> None:
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

    def _build_tool_parameters_schema(self, tool: Any) -> dict[str, Any] | None:
        """构造单个 MCP 工具的对象级参数 Schema。

        Args:
            tool: MCP SDK 返回的原始工具对象。

        Returns:
            dict[str, Any] | None: 参数 Schema。
        """

        parameters_schema = (
            dict(tool.inputSchema)
            if hasattr(tool, "inputSchema") and tool.inputSchema
            else {"type": "object", "properties": {}}
        )
        parameters_schema.pop("$schema", None)
        return parameters_schema

    def get_tool_specs(self) -> list[ToolSpec]:
        """获取全部已注册 MCP 工具的统一声明。

        Returns:
            list[ToolSpec]: MCP 工具声明列表。
        """

        tool_specs: list[ToolSpec] = []
        for server_name, conn in self._connections.items():
            for tool in conn.tools:
                if tool.name not in self._tool_to_server:
                    continue
                if self._tool_to_server[tool.name] != server_name:
                    continue

                parameters_schema = self._build_tool_parameters_schema(tool)
                brief_description = str(tool.description or f"来自 {server_name} 的 MCP 工具").strip()
                tool_specs.append(
                    ToolSpec(
                        name=str(tool.name),
                        brief_description=brief_description,
                        detailed_description=build_tool_detailed_description(
                            parameters_schema,
                            fallback_description=f"工具来源：MCP 服务 {server_name}。",
                        ),
                        parameters_schema=parameters_schema,
                        provider_name="mcp",
                        provider_type="mcp",
                        metadata={"server_name": server_name},
                    )
                )
        return tool_specs

    def get_openai_tools(self) -> list[dict[str, Any]]:
        """获取兼容旧模型层的 MCP 工具定义。

        Returns:
            list[dict[str, Any]]: OpenAI function tool 格式列表。
        """

        return [
            {
                "type": "function",
                "function": {
                    "name": tool_spec.name,
                    "description": tool_spec.build_llm_description(),
                    "parameters": tool_spec.parameters_schema or {"type": "object", "properties": {}},
                },
            }
            for tool_spec in self.get_tool_specs()
        ]

    # ──────── 工具调用 ────────

    def is_mcp_tool(self, tool_name: str) -> bool:
        """判断工具名是否为已注册的 MCP 工具。"""
        return tool_name in self._tool_to_server

    async def call_tool_invocation(self, invocation: ToolInvocation) -> ToolExecutionResult:
        """执行统一的 MCP 工具调用。

        Args:
            invocation: 统一工具调用请求。

        Returns:
            ToolExecutionResult: 统一工具执行结果。
        """

        tool_name = invocation.tool_name
        server_name = self._tool_to_server.get(tool_name)
        if not server_name or server_name not in self._connections:
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                error_message=f"MCP 工具 '{tool_name}' 未找到",
            )

        conn = self._connections[server_name]
        return await conn.call_tool(tool_name, invocation.arguments)

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """兼容旧接口，返回 MCP 工具的文本结果。

        Args:
            tool_name: 工具名称。
            arguments: 工具参数。

        Returns:
            str: 工具结果文本。
        """

        result = await self.call_tool_invocation(
            ToolInvocation(
                tool_name=tool_name,
                arguments=arguments,
            )
        )
        return result.get_history_content()

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

    async def close(self) -> None:
        """关闭所有 MCP 服务器连接。"""
        for conn in self._connections.values():
            await conn.close()
        self._connections.clear()
        self._tool_to_server.clear()
