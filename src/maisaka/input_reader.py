"""
MaiSaka - 异步输入读取器
基于后台线程的异步标准输入读取，通过 asyncio.Queue 传递给异步代码。
"""

import sys
import asyncio
import threading
from typing import Optional


class InputReader:
    """
    基于后台线程的异步标准输入读取器。

    使用单一守护线程持续读取 stdin，通过 asyncio.Queue 传递给异步代码。
    保证整个应用只有一个线程读 stdin，避免多线程竞争。
    支持带超时的读取，用于 LLM wait 工具。
    """

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    def start(self, loop: asyncio.AbstractEventLoop):
        """启动后台读取线程（仅首次调用生效）"""
        if self._thread is not None:
            return
        self._loop = loop
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        """后台线程：持续从 stdin 读取行"""
        try:
            while True:
                line = sys.stdin.readline()
                if not line:  # EOF
                    self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
                    break
                stripped = line.rstrip("\n").rstrip("\r")
                self._loop.call_soon_threadsafe(self._queue.put_nowait, stripped)
        except Exception:
            pass

    async def get_line(self, timeout: Optional[float] = None) -> Optional[str]:
        """
        异步获取下一行输入。

        Args:
            timeout: 超时秒数，None 表示无限等待

        Returns:
            输入的字符串，超时或 EOF 返回 None
        """
        try:
            if timeout is not None:
                return await asyncio.wait_for(self._queue.get(), timeout=timeout)
            return await self._queue.get()
        except asyncio.TimeoutError:
            return None
