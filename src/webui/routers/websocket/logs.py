"""WebSocket 日志推送路由兼容导出。"""

from src.webui.logs_ws import active_connections, broadcast_log, load_recent_logs, router, websocket_logs

__all__ = [
    "active_connections",
    "broadcast_log",
    "load_recent_logs",
    "router",
    "websocket_logs",
]
