"""
MaiSaka - MCP 配置加载与验证
从 mcp_config.json 读取 MCP 服务器定义，解析为结构化配置对象。

配置格式示例:
{
    "mcpServers": {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:/Users"],
            "env": {}
        },
        "remote-api": {
            "url": "http://localhost:8080/sse",
            "headers": {"Authorization": "Bearer xxx"}
        }
    }
}

- command + args: Stdio 传输（启动子进程）
- url: SSE 传输（连接远程服务器）
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from config import console


@dataclass
class MCPServerConfig:
    """单个 MCP 服务器配置。"""

    name: str

    # ── Stdio 传输 ──
    command: Optional[str] = None
    args: list[str] = field(default_factory=list)
    env: Optional[dict[str, str]] = None

    # ── SSE 传输 ──
    url: Optional[str] = None
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def transport_type(self) -> str:
        """返回传输类型: 'stdio' / 'sse' / 'unknown'。"""
        if self.command:
            return "stdio"
        if self.url:
            return "sse"
        return "unknown"


def load_mcp_config(config_path: str = "mcp_config.json") -> list[MCPServerConfig]:
    """
    从配置文件加载 MCP 服务器列表。

    Args:
        config_path: 配置文件路径

    Returns:
        解析后的 MCPServerConfig 列表；文件不存在或为空时返回空列表。
    """
    if not os.path.isfile(config_path):
        return []

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        console.print(f"[warning]⚠️ 读取 MCP 配置失败: {e}[/warning]")
        return []

    mcp_servers = data.get("mcpServers", {})
    if not isinstance(mcp_servers, dict):
        console.print("[warning]⚠️ mcp_config.json 中 mcpServers 格式无效[/warning]")
        return []

    configs: list[MCPServerConfig] = []
    for name, cfg in mcp_servers.items():
        if not isinstance(cfg, dict):
            console.print(f"[warning]⚠️ MCP 服务器 '{name}' 配置格式无效，已跳过[/warning]")
            continue

        server = MCPServerConfig(
            name=name,
            command=cfg.get("command"),
            args=cfg.get("args", []),
            env=cfg.get("env"),
            url=cfg.get("url"),
            headers=cfg.get("headers", {}),
        )

        if server.transport_type == "unknown":
            console.print(f"[warning]⚠️ MCP 服务器 '{name}' 缺少 command 或 url，已跳过[/warning]")
            continue

        configs.append(server)

    return configs
