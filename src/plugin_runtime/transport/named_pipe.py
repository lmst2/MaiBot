"""Windows Named Pipe 传输实现。

适用于 Windows 平台，使用 asyncio ProactorEventLoop 的 named pipe 支持。
"""

from typing import Any, Optional, Protocol, cast

import asyncio
import os
import re
import sys
import uuid

from .base import Connection, ConnectionHandler, TransportClient, TransportServer

_PIPE_PREFIX = "\\\\.\\pipe\\"
_DEFAULT_PIPE_PREFIX = "maibot-plugin"


class _NamedPipeEventLoop(Protocol):
    async def start_serving_pipe(self, protocol_factory: Any, address: str) -> list[Any]: ...

    async def create_pipe_connection(self, protocol_factory: Any, address: str) -> tuple[Any, Any]: ...

    def call_exception_handler(self, context: dict[str, Any]) -> None: ...

    def create_task(self, coro: Any) -> asyncio.Task[None]: ...


def _normalize_pipe_address(pipe_name: Optional[str] = None) -> str:
    if pipe_name and pipe_name.startswith(_PIPE_PREFIX):
        return pipe_name

    if pipe_name:
        sanitized_name = re.sub(r"[^0-9A-Za-z._-]+", "-", pipe_name).strip("-.")
    else:
        sanitized_name = f"{_DEFAULT_PIPE_PREFIX}-{os.getpid()}-{uuid.uuid4().hex[:8]}"

    if not sanitized_name:
        sanitized_name = f"{_DEFAULT_PIPE_PREFIX}-{os.getpid()}-{uuid.uuid4().hex[:8]}"

    return f"{_PIPE_PREFIX}{sanitized_name}"


class NamedPipeConnection(Connection):
    """基于 Windows Named Pipe 的连接。"""

    pass


class _NamedPipeServerProtocol(asyncio.StreamReaderProtocol):
    def __init__(self, handler: ConnectionHandler, loop: asyncio.AbstractEventLoop) -> None:
        self._reader = asyncio.StreamReader()
        super().__init__(self._reader)
        self._handler = handler
        self._loop = loop
        self._handler_task: Optional[asyncio.Task[None]] = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        super().connection_made(transport)
        writer = asyncio.StreamWriter(cast(asyncio.WriteTransport, transport), self, self._reader, self._loop)
        connection = NamedPipeConnection(self._reader, writer)
        self._handler_task = self._loop.create_task(self._run_handler(connection))
        self._handler_task.add_done_callback(self._on_handler_done)

    async def _run_handler(self, connection: NamedPipeConnection) -> None:
        try:
            await self._handler(connection)
        finally:
            await connection.close()

    def _on_handler_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        if exc := task.exception():
            self._loop.call_exception_handler(
                {
                    "message": "Named pipe 连接处理失败",
                    "exception": exc,
                    "protocol": self,
                }
            )


class NamedPipeTransportServer(TransportServer):
    """Windows Named Pipe 传输服务端。"""

    def __init__(self, pipe_name: Optional[str] = None) -> None:
        self._address = _normalize_pipe_address(pipe_name)
        self._servers: list[Any] = []

    async def start(self, handler: ConnectionHandler) -> None:
        if sys.platform != "win32":
            raise RuntimeError("Named pipe 仅支持 Windows")

        loop = asyncio.get_running_loop()
        if not hasattr(loop, "start_serving_pipe"):
            raise RuntimeError("当前事件循环不支持 Windows named pipe")
        pipe_loop = cast(_NamedPipeEventLoop, loop)

        self._servers = await pipe_loop.start_serving_pipe(
            lambda: _NamedPipeServerProtocol(handler, loop),
            self._address,
        )

    async def stop(self) -> None:
        for server in self._servers:
            server.close()
        self._servers.clear()
        await asyncio.sleep(0)

    def get_address(self) -> str:
        return self._address


class NamedPipeTransportClient(TransportClient):
    """Windows Named Pipe 传输客户端。"""

    def __init__(self, address: str) -> None:
        self._address = _normalize_pipe_address(address)

    async def connect(self) -> Connection:
        if sys.platform != "win32":
            raise RuntimeError("Named pipe 仅支持 Windows")

        loop = asyncio.get_running_loop()
        if not hasattr(loop, "create_pipe_connection"):
            raise RuntimeError("当前事件循环不支持 Windows named pipe")
        pipe_loop = cast(_NamedPipeEventLoop, loop)

        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        transport, _protocol = await pipe_loop.create_pipe_connection(lambda: protocol, self._address)
        writer = asyncio.StreamWriter(cast(asyncio.WriteTransport, transport), protocol, reader, loop)
        return NamedPipeConnection(reader, writer)