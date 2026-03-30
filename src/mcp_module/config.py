"""MCP 运行时配置转换。

负责将主程序官方配置中的 MCP 配置转换为运行时使用的结构化对象。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.official_configs import MCPConfig


@dataclass(slots=True)
class MCPServerRuntimeConfig:
    """单个 MCP 服务器的运行时配置。"""

    name: str
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def transport_type(self) -> str:
        """返回当前服务器的传输类型。

        Returns:
            str: ``stdio``、``sse`` 或 ``unknown``。
        """

        if self.command:
            return "stdio"
        if self.url:
            return "sse"
        return "unknown"


def build_mcp_server_runtime_configs(mcp_config: "MCPConfig") -> list[MCPServerRuntimeConfig]:
    """将官方 MCP 配置转换为运行时配置列表。

    Args:
        mcp_config: 主程序中的 MCP 官方配置对象。

    Returns:
        list[MCPServerRuntimeConfig]: 启用且配置完整的 MCP 服务器列表。
    """

    if not mcp_config.enable:
        return []

    runtime_configs: list[MCPServerRuntimeConfig] = []
    for server in mcp_config.servers:
        if not server.enabled:
            continue

        runtime_configs.append(
            MCPServerRuntimeConfig(
                name=server.name.strip(),
                command=server.command.strip(),
                args=[str(argument) for argument in server.args],
                env={str(key): str(value) for key, value in server.env.items()},
                url=server.url.strip(),
                headers={str(key): str(value) for key, value in server.headers.items()},
            )
        )

    return runtime_configs
