"""
MaiSaka - Debug Viewer 客户端
在独立命令行窗口中显示每次 LLM 调用的完整 Prompt。
通过 TCP socket 将数据发送给 debug_viewer.py 子进程。
"""

import json
import os
import socket
import struct
import subprocess
import sys
import time
from typing import Optional

from config import console


class DebugViewer:
    """
    在独立命令行窗口中显示每次 LLM 调用的完整 Prompt。

    通过 TCP socket 将数据发送给 debug_viewer.py 子进程。
    """

    def __init__(self, port: int = 19876):
        self._port = port
        self._conn: Optional[socket.socket] = None
        self._process: Optional[subprocess.Popen] = None

    def start(self):
        """启动调试窗口子进程并建立 TCP 连接。"""
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_viewer.py")

        try:
            self._process = subprocess.Popen(
                [sys.executable, script_path, str(self._port)],
                creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
            )
        except Exception as e:
            console.print(f"[warning]⚠️ 无法启动调试窗口: {e}[/warning]")
            return

        # 重试连接（等待子进程启动监听）
        for attempt in range(20):
            try:
                time.sleep(0.3)
                conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                conn.connect(("127.0.0.1", self._port))
                self._conn = conn
                console.print(f"[success]✓ 调试窗口已启动[/success] [muted](port {self._port})[/muted]")
                return
            except ConnectionRefusedError:
                conn.close()

        console.print("[warning]⚠️ 无法连接到调试窗口（超时）[/warning]")

    def send(self, label: str, messages: list, tools: Optional[list] = None, response: Optional[dict] = None):
        """发送一次 LLM 调用的完整 prompt 和响应到调试窗口。"""
        if not self._conn:
            return

        # 只在有响应时才发送（避免显示两次：请求中 + 完成响应）
        if response is None:
            return

        payload = {"label": label, "messages": messages}
        if tools:
            payload["tools"] = tools
        payload["response"] = response

        try:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            header = struct.pack(">I", len(data))
            self._conn.sendall(header + data)
        except Exception:
            # 连接断开时静默忽略
            self._conn = None

    def close(self):
        """关闭连接和子进程。"""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        if self._process:
            try:
                self._process.terminate()
            except Exception:
                pass
            self._process = None
