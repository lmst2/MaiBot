"""消息服务模块。"""

import re
import time
from datetime import datetime
from typing import Any, List, Optional, Tuple

from sqlmodel import col, select

from src.chat.message_receive.message import SessionMessage
from src.common.data_models.action_record_data_model import MaiActionRecord
from src.common.database.database import get_db_session
from src.common.database.database_model import ActionRecord, Images, ImageType
from src.common.message_repository import count_messages, find_messages
from src.common.utils.math_utils import translate_timestamp_to_human_readable
from src.common.utils.utils_action import ActionUtils
from src.chat.utils.utils import is_bot_self
from src.config.config import global_config


# =============================================================================
# 消息查询函数
# =============================================================================


def _build_time_range_filter(start_time: float, end_time: float) -> dict[str, Any]:
    return {
        "time": {
            "$gte": start_time,
            "$lte": end_time,
        }
    }


def _build_readable_line(
    message: SessionMessage,
    *,
    replace_bot_name: bool,
    timestamp_mode: Optional[str],
    show_message_id_prefix: bool,
) -> str:
    plain_text = (message.processed_plain_text or "").strip()
    if replace_bot_name and global_config.bot.nickname:
        plain_text = plain_text.replace(global_config.bot.nickname, "你")
    user_name = (
        message.message_info.user_info.user_cardname
        or message.message_info.user_info.user_nickname
        or message.message_info.user_info.user_id
    )
    prefix: List[str] = []
    if timestamp_mode:
        prefix.append(f"[{translate_timestamp_to_human_readable(message.timestamp.timestamp(), mode=timestamp_mode)}]")
    if show_message_id_prefix:
        prefix.append(f"[消息ID: {message.message_id}]")
    prefix.append(f"{user_name}说：")
    return " ".join(prefix) + plain_text


def _normalize_messages(messages: List[SessionMessage]) -> List[SessionMessage]:
    normalized: List[SessionMessage] = []
    for message in messages:
        if not message.processed_plain_text:
            message.processed_plain_text = message.display_message or ""
        normalized.append(message)
    return normalized


def get_messages_by_time(
    start_time: float, end_time: float, limit: int = 0, limit_mode: str = "latest", filter_mai: bool = False
) -> List[SessionMessage]:
    if not isinstance(start_time, (int, float)) or not isinstance(end_time, (int, float)):
        raise ValueError("start_time 和 end_time 必须是数字类型")
    if limit < 0:
        raise ValueError("limit 不能为负数")
    messages = find_messages(
        message_filter=_build_time_range_filter(start_time, end_time),
        limit=limit,
        limit_mode=limit_mode,
        filter_bot=filter_mai,
    )
    return _normalize_messages(messages)


def get_messages_by_time_in_chat(
    chat_id: str,
    start_time: float,
    end_time: float,
    limit: int = 0,
    limit_mode: str = "latest",
    filter_mai: bool = False,
    filter_command: bool = False,
    filter_intercept_message_level: Optional[int] = None,
) -> List[SessionMessage]:
    if not isinstance(start_time, (int, float)) or not isinstance(end_time, (int, float)):
        raise ValueError("start_time 和 end_time 必须是数字类型")
    if limit < 0:
        raise ValueError("limit 不能为负数")
    if not chat_id:
        raise ValueError("chat_id 不能为空")
    if not isinstance(chat_id, str):
        raise ValueError("chat_id 必须是字符串类型")
    messages = find_messages(
        message_filter={
            "chat_id": chat_id,
            **_build_time_range_filter(start_time, end_time),
        },
        limit=limit,
        limit_mode=limit_mode,
        filter_bot=filter_mai,
        filter_command=filter_command,
        filter_intercept_message_level=filter_intercept_message_level,
    )
    return _normalize_messages(messages)


def get_messages_by_time_in_chat_inclusive(
    chat_id: str,
    start_time: float,
    end_time: float,
    limit: int = 0,
    limit_mode: str = "latest",
    filter_mai: bool = False,
    filter_command: bool = False,
    filter_intercept_message_level: Optional[int] = None,
) -> List[SessionMessage]:
    if not isinstance(start_time, (int, float)) or not isinstance(end_time, (int, float)):
        raise ValueError("start_time 和 end_time 必须是数字类型")
    if limit < 0:
        raise ValueError("limit 不能为负数")
    if not chat_id:
        raise ValueError("chat_id 不能为空")
    if not isinstance(chat_id, str):
        raise ValueError("chat_id 必须是字符串类型")
    messages = find_messages(
        message_filter={
            "chat_id": chat_id,
            "time": {
                "$gte": start_time,
                "$lte": end_time,
            },
        },
        limit=limit,
        limit_mode=limit_mode,
        filter_bot=filter_mai,
        filter_command=filter_command,
        filter_intercept_message_level=filter_intercept_message_level,
    )
    return _normalize_messages(messages)


def get_messages_by_time_in_chat_for_users(
    chat_id: str,
    start_time: float,
    end_time: float,
    person_ids: List[str],
    limit: int = 0,
    limit_mode: str = "latest",
) -> List[SessionMessage]:
    if not isinstance(start_time, (int, float)) or not isinstance(end_time, (int, float)):
        raise ValueError("start_time 和 end_time 必须是数字类型")
    if limit < 0:
        raise ValueError("limit 不能为负数")
    if not chat_id:
        raise ValueError("chat_id 不能为空")
    if not isinstance(chat_id, str):
        raise ValueError("chat_id 必须是字符串类型")
    messages = find_messages(
        message_filter={
            "chat_id": chat_id,
            "time": {
                "$gte": start_time,
                "$lte": end_time,
            },
            "user_id": {"$in": person_ids},
        },
        limit=limit,
        limit_mode=limit_mode,
    )
    return _normalize_messages(messages)


def get_random_chat_messages(
    start_time: float, end_time: float, limit: int = 0, limit_mode: str = "latest", filter_mai: bool = False
) -> List[SessionMessage]:
    if not isinstance(start_time, (int, float)) or not isinstance(end_time, (int, float)):
        raise ValueError("start_time 和 end_time 必须是数字类型")
    if limit < 0:
        raise ValueError("limit 不能为负数")
    return get_messages_by_time(start_time, end_time, limit, limit_mode, filter_mai)


def get_messages_by_time_for_users(
    start_time: float, end_time: float, person_ids: List[str], limit: int = 0, limit_mode: str = "latest"
) -> List[SessionMessage]:
    if not isinstance(start_time, (int, float)) or not isinstance(end_time, (int, float)):
        raise ValueError("start_time 和 end_time 必须是数字类型")
    if limit < 0:
        raise ValueError("limit 不能为负数")
    messages = find_messages(
        message_filter={
            "time": {
                "$gte": start_time,
                "$lte": end_time,
            },
            "user_id": {"$in": person_ids},
        },
        limit=limit,
        limit_mode=limit_mode,
    )
    return _normalize_messages(messages)


def get_messages_before_time(timestamp: float, limit: int = 0, filter_mai: bool = False) -> List[SessionMessage]:
    if not isinstance(timestamp, (int, float)):
        raise ValueError("timestamp 必须是数字类型")
    if limit < 0:
        raise ValueError("limit 不能为负数")
    messages = find_messages(
        message_filter={"time": {"$lt": timestamp}},
        limit=limit,
        limit_mode="latest",
        filter_bot=filter_mai,
    )
    return _normalize_messages(messages)


def get_messages_before_time_in_chat(
    chat_id: str,
    timestamp: float,
    limit: int = 0,
    filter_mai: bool = False,
    filter_intercept_message_level: Optional[int] = None,
) -> List[SessionMessage]:
    if not isinstance(timestamp, (int, float)):
        raise ValueError("timestamp 必须是数字类型")
    if limit < 0:
        raise ValueError("limit 不能为负数")
    if not chat_id:
        raise ValueError("chat_id 不能为空")
    if not isinstance(chat_id, str):
        raise ValueError("chat_id 必须是字符串类型")
    messages = find_messages(
        message_filter={
            "chat_id": chat_id,
            "time": {"$lt": timestamp},
        },
        limit=limit,
        limit_mode="latest",
        filter_bot=filter_mai,
        filter_intercept_message_level=filter_intercept_message_level,
    )
    return _normalize_messages(messages)


def get_messages_before_time_for_users(
    timestamp: float, person_ids: List[str], limit: int = 0
) -> List[SessionMessage]:
    if not isinstance(timestamp, (int, float)):
        raise ValueError("timestamp 必须是数字类型")
    if limit < 0:
        raise ValueError("limit 不能为负数")
    messages = find_messages(
        message_filter={
            "time": {"$lt": timestamp},
            "user_id": {"$in": person_ids},
        },
        limit=limit,
        limit_mode="latest",
    )
    return _normalize_messages(messages)


def get_recent_messages(
    chat_id: str, hours: float = 24.0, limit: int = 100, limit_mode: str = "latest", filter_mai: bool = False
) -> List[SessionMessage]:
    if not isinstance(hours, (int, float)) or hours < 0:
        raise ValueError("hours 不能是负数")
    if not isinstance(limit, int) or limit < 0:
        raise ValueError("limit 必须是非负整数")
    if not chat_id:
        raise ValueError("chat_id 不能为空")
    if not isinstance(chat_id, str):
        raise ValueError("chat_id 必须是字符串类型")
    now = time.time()
    start_time = now - hours * 3600
    return get_messages_by_time_in_chat(chat_id, start_time, now, limit, limit_mode, filter_mai)


# =============================================================================
# 消息计数函数
# =============================================================================


def count_new_messages(chat_id: str, start_time: float = 0.0, end_time: Optional[float] = None) -> int:
    if not isinstance(start_time, (int, float)):
        raise ValueError("start_time 必须是数字类型")
    if not chat_id:
        raise ValueError("chat_id 不能为空")
    if not isinstance(chat_id, str):
        raise ValueError("chat_id 必须是字符串类型")
    message_filter: dict[str, Any] = {
        "chat_id": chat_id,
        "time": {"$gt": start_time},
    }
    if end_time is not None:
        message_filter["time"]["$lte"] = end_time
    return count_messages(message_filter)


def count_new_messages_for_users(chat_id: str, start_time: float, end_time: float, person_ids: List[str]) -> int:
    if not isinstance(start_time, (int, float)) or not isinstance(end_time, (int, float)):
        raise ValueError("start_time 和 end_time 必须是数字类型")
    if not chat_id:
        raise ValueError("chat_id 不能为空")
    if not isinstance(chat_id, str):
        raise ValueError("chat_id 必须是字符串类型")
    return count_messages(
        {
            "chat_id": chat_id,
            "time": {"$gt": start_time, "$lte": end_time},
            "user_id": {"$in": person_ids},
        }
    )


# =============================================================================
# 消息格式化函数
# =============================================================================


def build_readable_messages(
    messages: List[SessionMessage],
    replace_bot_name: bool = True,
    timestamp_mode: str = "relative",
    read_mark: float = 0.0,
    truncate: bool = False,
    show_actions: bool = False,
) -> str:
    normalized_messages = _normalize_messages(messages)
    lines: List[str] = []
    unread_mark_added = False
    for message in normalized_messages:
        if read_mark and not unread_mark_added and message.timestamp.timestamp() > read_mark:
            lines.append("--- 以上消息是你已经看过，请关注以下未读的新消息 ---")
            unread_mark_added = True
        line = _build_readable_line(
            message,
            replace_bot_name=replace_bot_name,
            timestamp_mode=timestamp_mode,
            show_message_id_prefix=False,
        )
        if truncate and len(line) > 200:
            line = f"{line[:200]}......（内容太长了）"
        lines.append(line)
    if show_actions and normalized_messages:
        action_lines = build_readable_actions(
            get_actions_by_timestamp_with_chat(
                normalized_messages[0].session_id,
                normalized_messages[0].timestamp.timestamp(),
                normalized_messages[-1].timestamp.timestamp(),
            )
        )
        if action_lines:
            lines.append(action_lines)
    return "\n".join(lines)


def build_readable_messages_to_str(
    messages: List[SessionMessage],
    replace_bot_name: bool = True,
    timestamp_mode: str = "relative",
    read_mark: float = 0.0,
    truncate: bool = False,
    show_actions: bool = False,
) -> str:
    return build_readable_messages(messages, replace_bot_name, timestamp_mode, read_mark, truncate, show_actions)


def build_readable_messages_with_id(
    messages: List[SessionMessage],
    replace_bot_name: bool = True,
    timestamp_mode: str = "relative",
    read_mark: float = 0.0,
    truncate: bool = False,
    show_actions: bool = False,
) -> Tuple[str, List[Tuple[str, SessionMessage]]]:
    normalized_messages = _normalize_messages(messages)
    lines: List[str] = []
    message_id_list: List[Tuple[str, SessionMessage]] = []
    unread_mark_added = False
    for message in normalized_messages:
        if read_mark and not unread_mark_added and message.timestamp.timestamp() > read_mark:
            lines.append("--- 以上消息是你已经看过，请关注以下未读的新消息 ---")
            unread_mark_added = True
        line = _build_readable_line(
            message,
            replace_bot_name=replace_bot_name,
            timestamp_mode=timestamp_mode,
            show_message_id_prefix=True,
        )
        if truncate and len(line) > 200:
            line = f"{line[:200]}......（内容太长了）"
        lines.append(line)
        message_id_list.append((message.message_id, message))
    if show_actions and normalized_messages:
        action_lines = build_readable_actions(
            get_actions_by_timestamp_with_chat(
                normalized_messages[0].session_id,
                normalized_messages[0].timestamp.timestamp(),
                normalized_messages[-1].timestamp.timestamp(),
            )
        )
        if action_lines:
            lines.append(action_lines)
    return "\n".join(lines), message_id_list


async def build_readable_messages_with_details(
    messages: List[SessionMessage],
    replace_bot_name: bool = True,
    timestamp_mode: str = "relative",
    truncate: bool = False,
) -> Tuple[str, List[Tuple[float, str, str]]]:
    normalized_messages = _normalize_messages(messages)
    message_list = [
        (
            message.timestamp.timestamp(),
            message.message_info.user_info.user_id,
            message.processed_plain_text or "",
        )
        for message in normalized_messages
    ]
    return build_readable_messages(normalized_messages, replace_bot_name, timestamp_mode, truncate=truncate), message_list


async def get_person_ids_from_messages(messages: List[Any]) -> List[str]:
    person_ids: List[str] = []
    for message in messages:
        if isinstance(message, SessionMessage):
            person_ids.append(message.message_info.user_info.user_id)
        elif isinstance(message, dict) and (user_id := message.get("user_id")):
            person_ids.append(str(user_id))
    return person_ids


# =============================================================================
# 消息过滤函数
# =============================================================================


def filter_mai_messages(messages: List[SessionMessage]) -> List[SessionMessage]:
    """从消息列表中移除麦麦的消息"""
    return [
        msg
        for msg in messages
        if not is_bot_self(msg.platform, msg.message_info.user_info.user_id)
    ]


def get_raw_msg_by_timestamp(
    timestamp_start: float,
    timestamp_end: float,
    limit: int = 0,
    limit_mode: str = "latest",
) -> List[SessionMessage]:
    return get_messages_by_time(timestamp_start, timestamp_end, limit, limit_mode)


def get_raw_msg_by_timestamp_with_chat(
    chat_id: str,
    timestamp_start: float,
    timestamp_end: float,
    limit: int = 0,
    limit_mode: str = "latest",
    filter_bot: bool = False,
    filter_command: bool = False,
    filter_intercept_message_level: Optional[int] = None,
) -> List[SessionMessage]:
    return get_messages_by_time_in_chat(
        chat_id,
        timestamp_start,
        timestamp_end,
        limit,
        limit_mode,
        filter_bot,
        filter_command,
        filter_intercept_message_level,
    )


def get_raw_msg_by_timestamp_with_chat_inclusive(
    chat_id: str,
    timestamp_start: float,
    timestamp_end: float,
    limit: int = 0,
    limit_mode: str = "latest",
    filter_bot: bool = False,
    filter_command: bool = False,
    filter_intercept_message_level: Optional[int] = None,
) -> List[SessionMessage]:
    return get_messages_by_time_in_chat_inclusive(
        chat_id,
        timestamp_start,
        timestamp_end,
        limit,
        limit_mode,
        filter_bot,
        filter_command,
        filter_intercept_message_level,
    )


def get_raw_msg_by_timestamp_with_chat_users(
    chat_id: str,
    timestamp_start: float,
    timestamp_end: float,
    person_ids: List[str],
    limit: int = 0,
    limit_mode: str = "latest",
) -> List[SessionMessage]:
    return get_messages_by_time_in_chat_for_users(chat_id, timestamp_start, timestamp_end, person_ids, limit, limit_mode)


def get_raw_msg_by_timestamp_with_users(
    timestamp_start: float,
    timestamp_end: float,
    person_ids: List[str],
    limit: int = 0,
    limit_mode: str = "latest",
) -> List[SessionMessage]:
    return get_messages_by_time_for_users(timestamp_start, timestamp_end, person_ids, limit, limit_mode)


def get_raw_msg_before_timestamp(timestamp: float, limit: int = 0) -> List[SessionMessage]:
    return get_messages_before_time(timestamp, limit)


def get_raw_msg_before_timestamp_with_chat(
    chat_id: str,
    timestamp: float,
    limit: int = 0,
    filter_intercept_message_level: Optional[int] = None,
) -> List[SessionMessage]:
    return get_messages_before_time_in_chat(chat_id, timestamp, limit, False, filter_intercept_message_level)


def get_raw_msg_before_timestamp_with_users(timestamp: float, person_ids: List[str], limit: int = 0) -> List[SessionMessage]:
    return get_messages_before_time_for_users(timestamp, person_ids, limit)


def get_raw_msg_by_timestamp_random(
    timestamp_start: float,
    timestamp_end: float,
    limit: int = 0,
    limit_mode: str = "latest",
) -> List[SessionMessage]:
    return get_random_chat_messages(timestamp_start, timestamp_end, limit, limit_mode)


def get_actions_by_timestamp_with_chat(chat_id: str, timestamp_start: float, timestamp_end: float) -> List[MaiActionRecord]:
    with get_db_session() as session:
        statement = (
            select(ActionRecord)
            .where(col(ActionRecord.session_id) == chat_id)
            .where(col(ActionRecord.timestamp) >= datetime.fromtimestamp(timestamp_start))
            .where(col(ActionRecord.timestamp) <= datetime.fromtimestamp(timestamp_end))
            .order_by(col(ActionRecord.timestamp))
        )
        return [MaiActionRecord.from_db_instance(item) for item in session.exec(statement).all()]


def build_readable_actions(actions: List[MaiActionRecord], timestamp_mode: str = "relative") -> str:
    return ActionUtils.build_readable_action_records(actions, timestamp_mode)


def replace_user_references(text: str, platform: str, replace_bot_name: bool = False) -> str:
    del platform
    if not text:
        return text

    def _replace(match: re.Match[str]) -> str:
        prefix = match.group(1) or ""
        user_name = match.group(2)
        if replace_bot_name and user_name == global_config.bot.nickname:
            user_name = "你"
        return f"{prefix}{user_name}"

    text = re.sub(r"(回复|@)?<([^:<>]+):[^<>]+>", _replace, text)
    return text


def translate_pid_to_description(pid: str) -> str:
    with get_db_session() as session:
        statement = (
            select(Images).where((col(Images.id) == int(pid)) & (col(Images.image_type) == ImageType.IMAGE))
            if pid.isdigit()
            else None
        )
        image = session.exec(statement).first() if statement is not None else None
    description = ""
    if image and image.description and image.description.strip():
        description = image.description.strip()
    else:
        description = "[图片]"
    return description
