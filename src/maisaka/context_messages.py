"""Maisaka 内部上下文消息抽象。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from io import BytesIO
from typing import Optional
import base64

from PIL import Image as PILImage

from src.chat.message_receive.message import SessionMessage
from src.common.data_models.message_component_data_model import EmojiComponent, ImageComponent, MessageSequence, TextComponent
from src.llm_models.payload_content.message import Message, MessageBuilder, RoleType
from src.llm_models.payload_content.tool_option import ToolCall


def _guess_image_format(image_bytes: bytes) -> Optional[str]:
    if not image_bytes:
        return None

    try:
        with PILImage.open(BytesIO(image_bytes)) as image:
            return image.format.lower() if image.format else None
    except Exception:
        return None


def _build_binary_component_type_text(component: EmojiComponent | ImageComponent) -> str:
    """为图片类消息组件构造显式的消息类型标记。"""
    if isinstance(component, EmojiComponent):
        return "[消息类型]表情包"
    return "[消息类型]图片"


def _build_message_from_sequence(
    role: RoleType,
    message_sequence: MessageSequence,
    fallback_text: str,
    *,
    tool_call_id: Optional[str] = None,
    tool_calls: Optional[list[ToolCall]] = None,
) -> Optional[Message]:
    """根据消息片段构造统一 LLM 消息。"""
    builder = MessageBuilder().set_role(role)
    if role == RoleType.Assistant and tool_calls:
        builder.set_tool_calls(tool_calls)
    if role == RoleType.Tool and tool_call_id:
        builder.add_tool_call(tool_call_id)

    has_content = False
    for component in message_sequence.components:
        if isinstance(component, TextComponent):
            if component.text:
                builder.add_text_content(component.text)
                has_content = True
            continue

        if isinstance(component, (EmojiComponent, ImageComponent)):
            image_format = _guess_image_format(component.binary_data)
            if image_format and component.binary_data:
                builder.add_text_content(_build_binary_component_type_text(component))
                builder.add_image_content(image_format, base64.b64encode(component.binary_data).decode("utf-8"))
                has_content = True
                continue

            if component.content:
                builder.add_text_content(component.content)
                has_content = True

    if not has_content and fallback_text:
        builder.add_text_content(fallback_text)
        has_content = True

    if not has_content and not (role == RoleType.Assistant and tool_calls):
        return None
    return builder.build()


class ReferenceMessageType(str, Enum):
    """参考消息类型。"""

    CUSTOM = "custom"
    JARGON = "jargon"
    KNOWLEDGE = "knowledge"
    MEMORY = "memory"
    TOOL_HINT = "tool_hint"


class LLMContextMessage(ABC):
    """Maisaka 内部用于组织 LLM 上下文的统一消息抽象。"""

    timestamp: datetime

    @property
    @abstractmethod
    def role(self) -> str:
        """返回 LLM 消息角色。"""

    @property
    @abstractmethod
    def processed_plain_text(self) -> str:
        """返回可读的纯文本内容。"""

    @property
    def count_in_context(self) -> bool:
        """是否占用普通 user/assistant 上下文窗口。"""
        return True

    @property
    def source(self) -> str:
        """返回消息来源。"""
        return self.__class__.__name__

    @abstractmethod
    def to_llm_message(self) -> Optional[Message]:
        """转换为统一 LLM 消息。"""

    def consume_once(self) -> bool:
        """消费一次生命周期，返回是否继续保留。"""
        return True


@dataclass(slots=True)
class SessionBackedMessage(LLMContextMessage):
    """真实会话上下文消息。"""

    raw_message: MessageSequence
    visible_text: str
    timestamp: datetime
    message_id: Optional[str] = None
    original_message: Optional[SessionMessage] = None
    source_kind: str = "user"

    @property
    def role(self) -> str:
        return RoleType.User.value

    @property
    def processed_plain_text(self) -> str:
        return self.visible_text

    @property
    def source(self) -> str:
        return self.source_kind

    def to_llm_message(self) -> Optional[Message]:
        return _build_message_from_sequence(
            RoleType.User,
            self.raw_message,
            self.processed_plain_text,
        )

    @classmethod
    def from_session_message(
        cls,
        session_message: SessionMessage,
        *,
        raw_message: MessageSequence,
        visible_text: str,
        source_kind: str = "user",
    ) -> "SessionBackedMessage":
        """从真实 SessionMessage 构造上下文消息。"""
        return cls(
            raw_message=raw_message,
            visible_text=visible_text,
            timestamp=session_message.timestamp,
            message_id=session_message.message_id,
            original_message=session_message,
            source_kind=source_kind,
        )


@dataclass(slots=True)
class ReferenceMessage(LLMContextMessage):
    """参考消息。"""

    content: str
    timestamp: datetime
    reference_type: ReferenceMessageType = ReferenceMessageType.CUSTOM
    remaining_uses_value: Optional[int] = 1
    display_prefix: str = "[参考消息]"

    @property
    def role(self) -> str:
        return RoleType.User.value

    @property
    def processed_plain_text(self) -> str:
        return f"{self.display_prefix}\n{self.content}".strip()

    @property
    def count_in_context(self) -> bool:
        return False

    @property
    def source(self) -> str:
        return self.reference_type.value

    def to_llm_message(self) -> Optional[Message]:
        message_sequence = MessageSequence([TextComponent(self.processed_plain_text)])
        return _build_message_from_sequence(RoleType.User, message_sequence, self.processed_plain_text)

    def consume_once(self) -> bool:
        if self.remaining_uses_value is None:
            return True

        self.remaining_uses_value -= 1
        return self.remaining_uses_value > 0


@dataclass(slots=True)
class AssistantMessage(LLMContextMessage):
    """内部 assistant 消息。"""

    content: str
    timestamp: datetime
    tool_calls: list[ToolCall] = field(default_factory=list)
    source_kind: str = "assistant"

    @property
    def role(self) -> str:
        return RoleType.Assistant.value

    @property
    def processed_plain_text(self) -> str:
        return self.content

    @property
    def count_in_context(self) -> bool:
        return self.source_kind != "perception"

    @property
    def source(self) -> str:
        return self.source_kind

    def to_llm_message(self) -> Optional[Message]:
        message_sequence = MessageSequence([])
        if self.content:
            message_sequence.text(self.content)
        return _build_message_from_sequence(
            RoleType.Assistant,
            message_sequence,
            self.content,
            tool_calls=self.tool_calls or None,
        )


@dataclass(slots=True)
class ToolResultMessage(LLMContextMessage):
    """工具返回结果消息。"""

    content: str
    timestamp: datetime
    tool_call_id: str
    tool_name: str = ""
    success: bool = True

    @property
    def role(self) -> str:
        return RoleType.Tool.value

    @property
    def processed_plain_text(self) -> str:
        return self.content

    @property
    def count_in_context(self) -> bool:
        return False

    @property
    def source(self) -> str:
        return self.tool_name or "tool"

    def to_llm_message(self) -> Optional[Message]:
        message_sequence = MessageSequence([TextComponent(self.content)])
        return _build_message_from_sequence(
            RoleType.Tool,
            message_sequence,
            self.content,
            tool_call_id=self.tool_call_id,
        )
