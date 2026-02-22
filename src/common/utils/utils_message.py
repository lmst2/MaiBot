from maim_message import MessageBase, Seg
from typing import List, Tuple, Optional

import base64
import hashlib
import msgpack
import re

from src.common.data_models.message_component_data_model import (
    MessageSequence,
    StandardMessageComponents,
    TextComponent,
    ImageComponent,
    EmojiComponent,
    VoiceComponent,
    AtComponent,
    ReplyComponent,
    DictComponent,
)
from src.config.config import global_config


class MessageUtils:
    @staticmethod
    def from_db_record_msg_to_MaiSeq(raw_content: bytes) -> MessageSequence:
        unpacked_data = msgpack.unpackb(raw_content)
        return MessageSequence.from_dict(unpacked_data)

    @staticmethod
    def from_MaiSeq_to_db_record_msg(msg: MessageSequence) -> bytes:
        dict_representation = msg.to_dict()
        return msgpack.packb(dict_representation)  # type: ignore

    @staticmethod
    def from_maim_message_segments_to_MaiSeq(message: "MessageBase") -> MessageSequence:
        """从maim_message.MessageBase.message_segment转换为MessageSequence"""
        raw_msg_seq = message.message_segment
        components: List[StandardMessageComponents] = []
        if not raw_msg_seq:
            return MessageSequence(components)
        if raw_msg_seq.type == "seglist":
            assert isinstance(raw_msg_seq.data, list), "seglist类型的message_segment数据应该是一个列表"
            components.extend(MessageUtils._parse_maim_message_segment_to_component(item) for item in raw_msg_seq.data)
        elif raw_msg_seq.type in {"text", "image", "emoji", "voice", "at", "reply"}:
            components.append(MessageUtils._parse_maim_message_segment_to_component(raw_msg_seq))
        else:
            raise NotImplementedError(f"暂时不支持的消息片段类型: {raw_msg_seq.type}")
        return MessageSequence(components)

    @staticmethod
    async def from_MaiSeq_to_maim_message_segments(msg_seq: MessageSequence) -> List[Seg]:
        """从MessageSequence转换为maim_message.MessageBase.message_segment格式的列表"""
        segments = []
        for component in msg_seq.components:
            if isinstance(component, DictComponent):
                seg = Seg(type="dict", data=component.data)  # type: ignore
            else:
                seg = await component.to_seg()
            segments.append(seg)
        return segments

    @staticmethod
    def _parse_maim_message_segment_to_component(seg: Seg) -> "StandardMessageComponents":
        if seg.type == "text":
            assert isinstance(seg.data, str), "text类型的seg数据应该是字符串"
            return TextComponent(text=seg.data)
        elif seg.type == "image":
            assert isinstance(seg.data, str), "image类型的seg数据应该是base64字符串"
            image_bytes = base64.b64decode(seg.data)
            binary_hash = hashlib.md5(image_bytes).hexdigest()
            return ImageComponent(binary_hash=binary_hash, binary_data=image_bytes)
        elif seg.type == "emoji":
            assert isinstance(seg.data, str), "emoji类型的seg数据应该是base64字符串"
            emoji_bytes = base64.b64decode(seg.data)
            binary_hash = hashlib.md5(emoji_bytes).hexdigest()
            return EmojiComponent(binary_hash=binary_hash, binary_data=emoji_bytes)
        elif seg.type == "voice":
            assert isinstance(seg.data, str), "voice类型的seg数据应该是base64字符串"
            voice_bytes = base64.b64decode(seg.data)
            binary_hash = hashlib.md5(voice_bytes).hexdigest()
            return VoiceComponent(binary_hash=binary_hash, binary_data=voice_bytes)
        elif seg.type == "at":
            assert isinstance(seg.data, str), "at类型的seg数据应该是字符串"
            return AtComponent(target_user_id=seg.data)
        elif seg.type == "reply":
            assert isinstance(seg.data, str), "reply类型的seg数据应该是字符串"
            return ReplyComponent(target_message_id=seg.data)
        else:
            raise NotImplementedError(f"暂时不支持的消息片段类型: {seg.type}")

    @staticmethod
    def check_ban_words(text: str) -> Tuple[bool, Optional[str]]:
        """检查消息是否包含过滤词

        Args:
            text: 待检查的文本

        Returns:
            bool: 是否包含过滤词
        """
        if not text:
            return False, None
        return next(
            ((True, word) for word in global_config.message_receive.ban_words if word in text),
            (False, None),
        )

    @staticmethod
    def check_ban_regex(text: str) -> Tuple[bool, Optional[str]]:
        """检查消息是否匹配过滤正则表达式

        Args:
            text: 待检查的文本
            chat: 聊天对象
            userinfo: 用户信息

        Returns:
            bool: 是否匹配过滤正则
        """
        # 检查text是否为None或空字符串
        if not text:
            return False, None
        return next(
            ((True, pattern) for pattern in global_config.message_receive.ban_msgs_regex if re.search(pattern, text)),
            (False, None),
        )

    @staticmethod
    def calculate_session_id(platform: str, *, user_id: Optional[str] = None, group_id: Optional[str] = None) -> str:
        """计算会话ID"""
        if not user_id and not group_id:
            raise ValueError("UserID 或 GroupID 必须提供其一")
        if group_id:
            components = [platform, group_id]
        else:
            components = [platform, user_id, "private"]
        return hashlib.md5("_".join(components).encode()).hexdigest()
