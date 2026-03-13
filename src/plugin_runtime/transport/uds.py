"""Unix Domain Socket 传输实现

适用于 Linux / macOS 平台。
"""

from pathlib import Path
from typing import Optional

import asyncio
import os
import tempfile

from .base import Connection, ConnectionHandler, TransportClient, TransportServer


class UDSConnection(Connection):
    """基于 UDS 的连接"""

    pass  # 直接复用 Connection 基类的分帧读写


# Unix domain socket 路径的系统限制（sun_path 字段长度）
# Linux: 108 字节, macOS: 104 字节
_UDS_PATH_MAX = 104


class UDSTransportServer(TransportServer):
    """UDS 传输服务端"""

    def __init__(self, socket_path: Optional[str] = None) -> None:
        if socket_path is None:
            # 默认放在临时目录，使用 uuid 确保同一进程多实例不碰撞
            import uuid

            socket_path = os.path.join(
                tempfile.gettempdir(), f"maibot-plugin-{os.getpid()}-{uuid.uuid4().hex[:8]}.sock"
            )

            # 如果路径超出 UDS 限制，回退到更短的路径
            if len(socket_path.encode()) > _UDS_PATH_MAX:
                socket_path = os.path.join("/tmp", f"mb-{os.getpid()}-{uuid.uuid4().hex[:8]}.sock")

        if len(socket_path.encode()) > _UDS_PATH_MAX:
            raise OSError(f"UDS socket 路径过长 ({len(socket_path.encode())} > {_UDS_PATH_MAX} 字节): {socket_path}")

        self._socket_path = socket_path
        self._server: Optional[asyncio.AbstractServer] = None

    async def start(self, handler: ConnectionHandler) -> None:
        # 清理残留 socket 文件
        if os.path.exists(self._socket_path):
            os.unlink(self._socket_path)

        # 确保父目录存在
        Path(self._socket_path).parent.mkdir(parents=True, exist_ok=True)

        async def _on_connect(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            conn = UDSConnection(reader, writer)
            try:
                await handler(conn)
            finally:
                await conn.close()

        self._server = await asyncio.start_unix_server(_on_connect, path=self._socket_path)

        # 设置文件权限为仅当前用户可访问
        os.chmod(self._socket_path, 0o600)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        # 清理 socket 文件
        if os.path.exists(self._socket_path):
            os.unlink(self._socket_path)

    def get_address(self) -> str:
        return self._socket_path


class UDSTransportClient(TransportClient):
    """UDS 传输客户端"""

    def __init__(self, socket_path: str) -> None:
        self._socket_path = socket_path

    async def connect(self) -> Connection:
        reader, writer = await asyncio.open_unix_connection(self._socket_path)
        return UDSConnection(reader, writer)
