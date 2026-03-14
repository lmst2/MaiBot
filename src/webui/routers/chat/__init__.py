from fastapi import APIRouter

from .routes import router
from .support import ChatConnectionManager, WEBUI_CHAT_PLATFORM, chat_manager


def get_webui_chat_broadcaster() -> tuple[ChatConnectionManager, str]:
    """获取 WebUI 聊天广播器，供外部模块使用。"""
    return chat_manager, WEBUI_CHAT_PLATFORM


__all__ = [
    "ChatConnectionManager",
    "WEBUI_CHAT_PLATFORM",
    "chat_manager",
    "get_webui_chat_broadcaster",
    "router",
]