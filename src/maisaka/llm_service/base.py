"""
MaiSaka - LLM 服务数据结构与抽象接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


# ──────────────────── 数据结构 ────────────────────

@dataclass
class ModelInfo:
    """模型描述信息"""
    model_name: str
    base_url: str


@dataclass
class ToolCall:
    """工具调用信息"""
    id: str
    name: str
    arguments: dict


@dataclass
class ChatResponse:
    """LLM 对话循环单步响应"""
    content: Optional[str]
    tool_calls: List[ToolCall]
    raw_message: dict  # 可直接追加到对话历史的消息字典


# ──────────────────── 抽象接口 ────────────────────

class BaseLLMService(ABC):
    """
    LLM 服务抽象基类。
    所有 LLM 后端实现都应继承此类，并实现以下方法。
    """

    def set_extra_tools(self, tools: List[dict]) -> None:
        """
        设置额外的工具定义（如 MCP 工具），将与内置工具合并使用。

        Args:
            tools: OpenAI function calling 格式的工具定义列表
        """
        # 默认空实现，子类可覆盖
        pass

    @abstractmethod
    async def chat_loop_step(self, chat_history: List[dict]) -> ChatResponse:
        """
        执行对话循环的一步。

        发送当前对话历史，获取 LLM 响应（可能包含文本和/或工具调用）。
        调用方需要将 raw_message 追加到 chat_history，并根据 tool_calls 执行工具、
        将工具结果追加到 chat_history 后再次调用本方法。

        Args:
            chat_history: 对话历史（含 system / user / assistant / tool 消息）

        Returns:
            ChatResponse
        """
        ...

    @abstractmethod
    def build_chat_context(self, user_text: str) -> List[dict]:
        """根据用户初始输入，构建对话循环的初始上下文（system + user）。"""
        ...

    @abstractmethod
    async def analyze_timing(
        self, chat_history: List[dict], timing_info: str,
    ) -> str:
        """
        Timing 模块（含自我反思功能）：分析对话的时间维度信息和进行自我反思。

        评估对话已经持续多久、上次回复距今多长时间、建议等待时长、
        以及其他与时间节奏相关的考量。同时反思自己的回复逻辑，
        检查人设一致性、回复合理性和认知局限性。

        Args:
            chat_history: 当前对话历史（与主 Agent 完全一致的上下文）
            timing_info:  系统提供的精确时间戳信息（对话开始时间、各消息时间等）

        Returns:
            时间维度分析和自我反思的综合文本
        """
        ...

    @abstractmethod
    async def analyze_emotion(self, chat_history: List[dict]) -> str:
        """
        情商模块：分析对话对方（用户）的情绪状态和言语态度。

        接收与主 Agent 相同的上下文，返回一段简洁的情绪分析文本。
        该文本将被注入主 Agent 上下文，帮助主 Agent 更好地理解用户状态。

        Args:
            chat_history: 当前对话历史（与主 Agent 完全一致的上下文）

        Returns:
            情绪分析文本
        """
        ...

    @abstractmethod
    async def analyze_cognition(self, chat_history: List[dict]) -> str:
        """
        认知模块：分析对话对方（用户）的意图、认知状态和目的。

        接收与主 Agent 相同的上下文，返回一段简洁的认知分析文本。
        该文本将被注入主 Agent 上下文，帮助主 Agent 更好地理解用户意图。

        Args:
            chat_history: 当前对话历史（与主 Agent 完全一致的上下文）

        Returns:
            认知分析文本
        """
        ...

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """返回当前使用的模型信息。"""
        ...

    @abstractmethod
    async def summarize_context(self, context_messages: List[dict]) -> str:
        """
        上下文总结模块：对需要压缩的上下文进行总结。

        当对话历史过长时，对早期的对话内容进行总结。

        Args:
            context_messages: 需要总结的上下文消息列表

        Returns:
            总结后的文本内容
        """
        ...

    @abstractmethod
    async def analyze_knowledge_categories(
        self, context_messages: List[dict], categories_summary: str
    ) -> List[str]:
        """
        了解模块-分类分析：分析对话内容涉及哪些个人特征分类。

        在上下文裁切时触发，分析需要提取哪些分类的个人特征信息。

        Args:
            context_messages: 需要分析的上下文消息
            categories_summary: 所有分类的摘要信息

        Returns:
            涉及的分类编号列表
        """
        ...

    @abstractmethod
    async def extract_knowledge_for_category(
        self, context_messages: List[dict], category_id: str, category_name: str
    ) -> str:
        """
        了解模块-内容提取：从对话中提取指定分类的个人特征信息。

        为每个分类创建 subAgent，提取相关的个人特征内容。

        Args:
            context_messages: 需要分析的上下文消息
            category_id: 分类编号
            category_name: 分类名称

        Returns:
            提取的个人特征内容
        """
        ...

    @abstractmethod
    async def analyze_knowledge_need(
        self, chat_history: List[dict], categories_summary: str
    ) -> List[str]:
        """
        了解模块-需求分析：分析当前对话需要哪些个人特征信息。

        在每次对话前触发，分析需要检索哪些分类的了解内容。

        Args:
            chat_history: 当前对话历史
            categories_summary: 所有分类的摘要信息

        Returns:
            需要的分类编号列表
        """
        ...
