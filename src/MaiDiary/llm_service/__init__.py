"""
MaiSaka - LLM 服务包
提供抽象接口 (BaseLLMService) 和 OpenAI 兼容实现 (OpenAILLMService)。
"""

from .base import BaseLLMService, ChatResponse, ModelInfo, ToolCall
from .openai_impl import OpenAILLMService
from .utils import format_chat_history

__all__ = [
    "BaseLLMService",
    "ChatResponse",
    "ModelInfo",
    "ToolCall",
    "OpenAILLMService",
    "format_chat_history",
]
