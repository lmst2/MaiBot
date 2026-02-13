from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from maim_message import MessageBase
from typing import Optional

import json
from datetime import datetime

from src.common.database.database_model import Messages
from src.common.data_models.message_component_model import MessageSequence
from src.common.utils.utils_message import MessageUtils


class BaseMAIMessageModel(ABC):
    @classmethod
    @abstractmethod
    def from_db_instance(cls, message: "Messages"):
        raise NotImplementedError

    @abstractmethod
    def to_db_instance(self) -> "Messages":
        raise NotImplementedError

    @abstractmethod
    def from_maim_message(cls, message: MessageBase):
        raise NotImplementedError

    @abstractmethod
    def to_maim_message(self) -> MessageBase:
        raise NotImplementedError

    @abstractmethod
    def parse_message_segments(self):
        raise NotImplementedError


@dataclass
class UserInfo:
    user_id: str
    user_nickname: str
    user_cardname: Optional[str] = None


@dataclass
class GroupInfo:
    group_id: str
    group_name: str


@dataclass
class MessageInfo:
    user_info: UserInfo
    group_info: Optional[GroupInfo] = None
    additional_config: dict = field(default_factory=dict)


class MaiMessage(BaseMAIMessageModel):
    def __init__(self, message_id: str, timestamp: datetime):
        self.message_id: str = message_id
        self.timestamp: datetime = timestamp  # 时间戳
        self.initialized = False  # 用于标记是否已初始化其他属性
        self.platform: str  # 初始化后赋值

        # 定义其他属性
        self.message_info: MessageInfo  # 初始化后赋值
        self.is_mentioned: bool = False
        self.is_at: bool = False
        self.is_emoji: bool = False
        self.is_picture: bool = False
        self.is_command: bool = False
        self.is_notify: bool = False

        self.session_id: str
        self.reply_to: Optional[str] = None

        self.processed_plain_text: Optional[str] = None
        self.display_message: Optional[str] = None
        self.raw_message: MessageSequence

    @classmethod
    def from_db_instance(cls, message: "Messages") -> "MaiMessage":
        obj = cls(message_id=message.message_id, timestamp=message.timestamp)

        user_info = UserInfo(message.user_id, message.user_nickname, message.user_cardname)
        if message.group_id and message.group_name:
            group_info = GroupInfo(message.group_id, message.group_name)
        else:
            group_info = None
        obj.message_info = MessageInfo(
            user_info=user_info,
            group_info=group_info,
            additional_config=json.loads(message.additional_config) if message.additional_config else {},
        )

        obj.is_mentioned = message.is_mentioned
        obj.is_at = message.is_at
        obj.is_emoji = message.is_emoji
        obj.is_picture = message.is_picture
        obj.is_command = message.is_command
        obj.is_notify = message.is_notify
        obj.reply_to = message.reply_to
        obj.session_id = message.session_id
        obj.processed_plain_text = message.processed_plain_text
        obj.display_message = message.display_message
        obj.raw_message = MessageUtils.from_db_record_msg_to_MaiSeq(message.raw_content)
        return obj

    def to_db_instance(self) -> Messages:
        additional_config = (
            json.dumps(self.message_info.additional_config) if self.message_info.additional_config else None
        )
        return Messages(
            message_id=self.message_id,
            timestamp=self.timestamp,
            platform=self.platform,
            user_id=self.message_info.user_info.user_id,
            user_nickname=self.message_info.user_info.user_nickname,
            user_cardname=self.message_info.user_info.user_cardname,
            group_id=self.message_info.group_info.group_id if self.message_info.group_info else None,
            group_name=self.message_info.group_info.group_name if self.message_info.group_info else None,
            is_mentioned=self.is_mentioned,
            is_at=self.is_at,
            session_id=self.session_id,
            reply_to=self.reply_to,
            is_emoji=self.is_emoji,
            is_picture=self.is_picture,
            is_command=self.is_command,
            is_notify=self.is_notify,
            raw_content=MessageUtils.from_MaiSeq_to_db_record_msg(self.raw_message),
            processed_plain_text=self.processed_plain_text,
            display_message=self.display_message,
            additional_config=additional_config,
        )
