from datetime import datetime
from collections.abc import Mapping
from typing import cast

import json
import re
import traceback

from sqlmodel import col, select
from src.common.database.database import get_db_session
from src.common.database.database_model import Images, ImageType, Messages
from src.common.logger import get_logger
from src.common.data_models.message_component_model import MessageSequence, TextComponent
from src.common.utils.utils_message import MessageUtils
from .chat_stream import ChatStream
from .message import MessageRecv, MessageSending

logger = get_logger("message_storage")


class MessageStorage:
    @staticmethod
    def _coerce_str_list(value: object) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, tuple):
            return [str(item) for item in value]
        if isinstance(value, set):
            return [str(item) for item in value]
        if isinstance(value, str):
            return [value]
        return []

    @staticmethod
    def _get_str(mapping: Mapping[str, object], key: str, default: str = "") -> str:
        value = mapping.get(key)
        if value is None:
            return default
        return str(value)

    @staticmethod
    def _get_optional_str(mapping: Mapping[str, object], key: str) -> str | None:
        value = mapping.get(key)
        if value is None:
            return None
        return str(value)

    @staticmethod
    def _serialize_keywords(keywords: list[str] | None) -> str:
        """将关键词列表序列化为JSON字符串"""
        if isinstance(keywords, list):
            return json.dumps(keywords, ensure_ascii=False)
        return "[]"

    @staticmethod
    def _deserialize_keywords(keywords_str: str) -> list[str]:
        """将JSON字符串反序列化为关键词列表"""
        if not keywords_str:
            return []
        try:
            parsed = cast(object, json.loads(keywords_str))
        except (json.JSONDecodeError, TypeError):
            return []
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        if isinstance(parsed, str):
            return [parsed]
        return []

    @staticmethod
    async def store_message(message: MessageSending | MessageRecv, chat_stream: ChatStream) -> None:
        """存储消息到数据库"""
        try:
            # 通知消息不存储
            if isinstance(message, MessageRecv) and message.is_notify:
                logger.debug("通知消息，跳过存储")
                return

            pattern = r"<MainRule>.*?</MainRule>|<schedule>.*?</schedule>|<UserMessage>.*?</UserMessage>"

            # print(message)

            processed_plain_text = message.processed_plain_text

            # print(processed_plain_text)

            if processed_plain_text:
                processed_plain_text = MessageStorage.replace_image_descriptions(processed_plain_text)
                filtered_processed_plain_text = re.sub(pattern, "", processed_plain_text, flags=re.DOTALL)
            else:
                filtered_processed_plain_text = ""

            if isinstance(message, MessageSending):
                display_message = message.display_message
                if display_message:
                    filtered_display_message = re.sub(pattern, "", display_message, flags=re.DOTALL)
                else:
                    filtered_display_message = ""
                interest_value = 0
                is_mentioned = False
                is_at = False
                reply_probability_boost = 0.0
                reply_to = message.reply_to
                priority_mode = ""
                priority_info = {}
                is_emoji = False
                is_picture = False
                is_notify = False
                is_command = False
                key_words = ""
                key_words_lite = ""
                selected_expressions = message.selected_expressions
                intercept_message_level = 0
            else:
                filtered_display_message = ""
                interest_value = message.interest_value
                is_mentioned = message.is_mentioned
                is_at = message.is_at
                reply_probability_boost = message.reply_probability_boost
                reply_to = ""
                priority_mode = message.priority_mode
                priority_info = message.priority_info
                is_emoji = message.is_emoji
                is_picture = message.is_picid
                is_notify = message.is_notify
                is_command = message.is_command
                intercept_message_level = getattr(message, "intercept_message_level", 0)
                # 序列化关键词列表为JSON字符串
                key_words = MessageStorage._serialize_keywords(MessageStorage._coerce_str_list(message.key_words))
                key_words_lite = MessageStorage._serialize_keywords(
                    MessageStorage._coerce_str_list(message.key_words_lite)
                )
                selected_expressions = ""

            chat_info_dict = cast(dict[str, object], chat_stream.to_dict())
            if message.message_info.user_info is None:
                raise ValueError("message.user_info is required")
            user_info_dict = cast(dict[str, object], message.message_info.user_info.to_dict())

            # message_id 现在是 TextField，直接使用字符串值
            msg_id = message.message_info.message_id or ""

            # 安全地获取 group_info, 如果为 None 则视为空字典
            group_info_from_chat = cast(dict[str, object], chat_info_dict.get("group_info") or {})

            additional_config: dict[str, object] = dict(message.message_info.additional_config or {})
            additional_config.update(
                {
                    "interest_value": interest_value,
                    "priority_mode": priority_mode,
                    "priority_info": priority_info,
                    "reply_probability_boost": reply_probability_boost,
                    "intercept_message_level": intercept_message_level,
                    "key_words": key_words,
                    "key_words_lite": key_words_lite,
                    "selected_expressions": selected_expressions,
                    "is_picid": is_picture,
                }
            )
            processed_text_for_raw = filtered_processed_plain_text or filtered_display_message or ""
            raw_sequence = MessageSequence([TextComponent(processed_text_for_raw)] if processed_text_for_raw else [])
            raw_content = MessageUtils.from_MaiSeq_to_db_record_msg(raw_sequence)

            timestamp_value = message.message_info.time
            if timestamp_value is None:
                raise ValueError("message.message_info.time is required")
            db_message = Messages(
                message_id=str(msg_id),
                timestamp=datetime.fromtimestamp(float(timestamp_value)),
                platform=MessageStorage._get_str(chat_info_dict, "platform"),
                user_id=MessageStorage._get_str(user_info_dict, "user_id"),
                user_nickname=MessageStorage._get_str(user_info_dict, "user_nickname"),
                user_cardname=MessageStorage._get_optional_str(user_info_dict, "user_cardname"),
                group_id=MessageStorage._get_optional_str(group_info_from_chat, "group_id"),
                group_name=MessageStorage._get_optional_str(group_info_from_chat, "group_name"),
                is_mentioned=bool(is_mentioned),
                is_at=bool(is_at),
                session_id=chat_stream.stream_id,
                reply_to=reply_to,
                is_emoji=is_emoji,
                is_picture=is_picture,
                is_command=is_command,
                is_notify=is_notify,
                raw_content=raw_content,
                processed_plain_text=filtered_processed_plain_text,
                display_message=filtered_display_message,
                additional_config=json.dumps(additional_config, ensure_ascii=False),
            )
            with get_db_session() as session:
                session.add(db_message)
        except Exception:
            logger.exception("存储消息失败")
            logger.error(f"消息：{message}")
            traceback.print_exc()

    # 如果需要其他存储相关的函数，可以在这里添加
    @staticmethod
    def update_message(mmc_message_id: str | None, qq_message_id: str | None) -> bool:
        """实时更新数据库的自身发送消息ID"""
        try:
            if not qq_message_id:
                logger.info("消息不存在message_id，无法更新")
                return False
            with get_db_session() as session:
                statement = (
                    select(Messages)
                    .where(col(Messages.message_id) == mmc_message_id)
                    .order_by(col(Messages.timestamp).desc())
                    .limit(1)
                )
                matched_message = session.exec(statement).first()
                if matched_message:
                    matched_message.message_id = qq_message_id
                    session.add(matched_message)
                    logger.debug(f"更新消息ID成功: {matched_message.message_id} -> {qq_message_id}")
                    return True
            logger.debug("未找到匹配的消息")
            return False

        except Exception as e:
            logger.error(f"更新消息ID失败: {e}")
            return False

    @staticmethod
    def replace_image_descriptions(text: str) -> str:
        """将[图片：描述]替换为[picid:image_id]"""
        # 先检查文本中是否有图片标记
        pattern = r"\[图片：([^\]]+)\]"
        matches = re.findall(pattern, text)

        if not matches:
            logger.debug("文本中没有图片标记，直接返回原文本")
            return text

        def replace_match(match: re.Match[str]) -> str:
            description = match.group(1).strip()
            try:
                with get_db_session() as session:
                    statement = (
                        select(Images)
                        .where((col(Images.description) == description) & (col(Images.image_type) == ImageType.IMAGE))
                        .order_by(col(Images.record_time).desc())
                        .limit(1)
                    )
                    image_record = session.exec(statement).first()
                return f"[picid:{image_record.id}]" if image_record else match.group(0)
            except Exception:
                return match.group(0)

        return re.sub(r"\[图片：([^\]]+)\]", replace_match, text)
