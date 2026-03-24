"""
MaiSaka message adapters built on top of the main project's MaiMessage model.
"""

from copy import deepcopy
from datetime import datetime
from io import BytesIO
from typing import Optional
from uuid import uuid4
import base64
import re

from PIL import Image as PILImage

from src.common.data_models.mai_message_data_model import GroupInfo, MaiMessage, MessageInfo, UserInfo
from src.common.data_models.message_component_data_model import EmojiComponent, ImageComponent, MessageSequence, TextComponent
from src.config.config import global_config
from src.llm_models.payload_content.message import Message, MessageBuilder, RoleType
from src.llm_models.payload_content.tool_option import ToolCall

from .config import USER_NAME

MAISAKA_PLATFORM = "maisaka"
MAISAKA_SESSION_ID = "maisaka_cli"
MESSAGE_KIND_KEY = "maisaka_message_kind"
SOURCE_KEY = "maisaka_source"
LLM_ROLE_KEY = "maisaka_llm_role"
TOOL_CALL_ID_KEY = "maisaka_tool_call_id"
TOOL_CALLS_KEY = "maisaka_tool_calls"
SPEAKER_PREFIX_PATTERN = re.compile(
    r"^(?:(?P<timestamp>\d{2}:\d{2}:\d{2}))?(?:<mid:(?P<message_id>[^>]+)>)?\[(?P<speaker>[^\]]+)\](?P<content>.*)$",
    re.DOTALL,
)


def _build_user_info_for_role(role: str) -> UserInfo:
    if role == RoleType.User.value:
        return UserInfo(user_id="maisaka_user", user_nickname=USER_NAME, user_cardname=None)
    if role == RoleType.Tool.value:
        return UserInfo(user_id="maisaka_tool", user_nickname="tool", user_cardname=None)
    return UserInfo(
        user_id="maisaka_assistant",
        user_nickname=global_config.bot.nickname.strip() or "MaiSaka",
        user_cardname=None,
    )


def _serialize_tool_call(tool_call: ToolCall) -> dict:
    return {
        "call_id": tool_call.call_id,
        "func_name": tool_call.func_name,
        "args": tool_call.args or {},
    }


def _deserialize_tool_call(data: dict) -> ToolCall:
    return ToolCall(
        call_id=str(data.get("call_id", "")),
        func_name=str(data.get("func_name", "")),
        args=data.get("args", {}) or {},
    )


def build_message(
    role: str,
    content: str = "",
    *,
    message_kind: str = "normal",
    source: Optional[str] = None,
    tool_call_id: Optional[str] = None,
    tool_calls: Optional[list[ToolCall]] = None,
    timestamp: Optional[datetime] = None,
    message_id: Optional[str] = None,
    platform: str = MAISAKA_PLATFORM,
    session_id: str = MAISAKA_SESSION_ID,
    user_info: Optional[UserInfo] = None,
    group_info: Optional[GroupInfo] = None,
    raw_message: Optional[MessageSequence] = None,
    display_text: Optional[str] = None,
) -> MaiMessage:
    """Build a MaiMessage for the Maisaka session history."""
    resolved_timestamp = timestamp or datetime.now()
    resolved_role = role.value if isinstance(role, RoleType) else role
    message = MaiMessage(
        message_id=message_id or f"maisaka_{uuid4().hex}",
        timestamp=resolved_timestamp,
        platform=platform,
    )
    message.message_info = MessageInfo(
        user_info=user_info or _build_user_info_for_role(resolved_role),
        group_info=group_info,
        additional_config={
            LLM_ROLE_KEY: resolved_role,
            MESSAGE_KIND_KEY: message_kind,
            SOURCE_KEY: source or resolved_role,
            TOOL_CALL_ID_KEY: tool_call_id,
            TOOL_CALLS_KEY: [_serialize_tool_call(tool_call) for tool_call in (tool_calls or [])],
        },
    )
    message.session_id = session_id
    message.raw_message = raw_message if raw_message is not None else MessageSequence([])
    if raw_message is None and content:
        message.raw_message.text(content)
    visible_text = display_text if display_text is not None else content
    message.processed_plain_text = visible_text
    message.display_message = visible_text
    return message


def format_speaker_content(
    speaker_name: str,
    content: str,
    timestamp: Optional[datetime] = None,
    message_id: Optional[str] = None,
) -> str:
    """Format visible conversation content with an explicit speaker label."""
    time_prefix = timestamp.strftime("%H:%M:%S") if timestamp is not None else ""
    message_id_prefix = f"<mid:{message_id}>" if message_id else ""
    return f"{time_prefix}{message_id_prefix}[{speaker_name}]{content}"


def parse_speaker_content(content: str) -> tuple[Optional[str], str]:
    """Parse content formatted as [speaker]message."""
    match = SPEAKER_PREFIX_PATTERN.match(content or "")
    if not match:
        return None, content or ""
    return match.group("speaker"), match.group("content")


def clone_message_sequence(message_sequence: MessageSequence) -> MessageSequence:
    """Create a detached copy of a message sequence."""
    return MessageSequence([deepcopy(component) for component in message_sequence.components])


def build_visible_text_from_sequence(message_sequence: MessageSequence) -> str:
    """Extract visible text from a message sequence without forcing image descriptions."""
    parts: list[str] = []
    for component in message_sequence.components:
        if isinstance(component, TextComponent):
            parts.append(SPEAKER_PREFIX_PATTERN.sub(r"\g<timestamp>[\g<speaker>]\g<content>", component.text))
            continue

        if isinstance(component, EmojiComponent):
            parts.append("[表情包]")
            continue

        if isinstance(component, ImageComponent):
            parts.append("[图片]")
    return "".join(parts)


def _guess_image_format(image_bytes: bytes) -> Optional[str]:
    if not image_bytes:
        return None

    try:
        with PILImage.open(BytesIO(image_bytes)) as image:
            return image.format.lower() if image.format else None
    except Exception:
        return None


def get_message_text(message: MaiMessage) -> str:
    if message.processed_plain_text is not None:
        return message.processed_plain_text
    if message.display_message is not None:
        return message.display_message

    parts: list[str] = []
    for component in message.raw_message.components:
        text = getattr(component, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def get_message_role(message: MaiMessage) -> str:
    return str(message.message_info.additional_config.get(LLM_ROLE_KEY, RoleType.User.value))


def get_message_kind(message: MaiMessage) -> str:
    return str(message.message_info.additional_config.get(MESSAGE_KIND_KEY, "normal"))


def get_message_source(message: MaiMessage) -> str:
    return str(message.message_info.additional_config.get(SOURCE_KEY, get_message_role(message)))


def is_perception_message(message: MaiMessage) -> bool:
    return get_message_kind(message) == "perception"


def get_tool_call_id(message: MaiMessage) -> Optional[str]:
    value = message.message_info.additional_config.get(TOOL_CALL_ID_KEY)
    return str(value) if value else None


def get_tool_calls(message: MaiMessage) -> list[ToolCall]:
    raw_tool_calls = message.message_info.additional_config.get(TOOL_CALLS_KEY, [])
    if not isinstance(raw_tool_calls, list):
        return []
    return [_deserialize_tool_call(item) for item in raw_tool_calls if isinstance(item, dict)]


def remove_last_perception(messages: list[MaiMessage]) -> None:
    for index in range(len(messages) - 1, -1, -1):
        if is_perception_message(messages[index]):
            messages.pop(index)
            break


def to_llm_message(message: MaiMessage) -> Optional[Message]:
    role = get_message_role(message)
    tool_call_id = get_tool_call_id(message)
    tool_calls = get_tool_calls(message)

    if role == RoleType.System.value:
        role_type = RoleType.System
    elif role == RoleType.User.value:
        role_type = RoleType.User
    elif role == RoleType.Assistant.value:
        role_type = RoleType.Assistant
    elif role == RoleType.Tool.value:
        role_type = RoleType.Tool
    else:
        return None

    builder = MessageBuilder().set_role(role_type)
    if role_type == RoleType.Assistant and tool_calls:
        builder.set_tool_calls(tool_calls)
    if role_type == RoleType.Tool and tool_call_id:
        builder.add_tool_call(tool_call_id)

    has_content = False
    for component in message.raw_message.components:
        if isinstance(component, TextComponent):
            if component.text:
                builder.add_text_content(component.text)
                has_content = True
            continue

        if isinstance(component, (ImageComponent, EmojiComponent)):
            image_format = _guess_image_format(component.binary_data)
            if image_format and component.binary_data:
                builder.add_image_content(image_format, base64.b64encode(component.binary_data).decode("utf-8"))
                has_content = True
                continue

            if component.content:
                builder.add_text_content(component.content)
                has_content = True

    if not has_content:
        content = get_message_text(message)
        if content:
            builder.add_text_content(content)
    return builder.build()
