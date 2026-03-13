"""
聊天服务模块

提供聊天信息查询和管理的核心功能。
"""

from typing import Any, Callable, Dict, List, Optional
from enum import Enum

from src.chat.message_receive.chat_manager import BotChatSession, chat_manager as _chat_manager
from src.common.logger import get_logger

logger = get_logger("chat_service")


class SpecialTypes(Enum):
    """特殊枚举类型"""

    ALL_PLATFORMS = "all_platforms"


class ChatManager:
    """聊天管理器 - 负责聊天信息的查询和管理"""

    @staticmethod
    def _validate_platform(platform: Optional[str] | SpecialTypes) -> None:
        if not isinstance(platform, (str, SpecialTypes)):
            raise TypeError("platform 必须是字符串或是 SpecialTypes 枚举")

    @staticmethod
    def _match_platform(chat_stream: BotChatSession, platform: Optional[str] | SpecialTypes) -> bool:
        return platform == SpecialTypes.ALL_PLATFORMS or chat_stream.platform == platform

    @staticmethod
    def _get_streams(
        platform: Optional[str] | SpecialTypes = "qq", is_group_session: Optional[bool] = None
    ) -> List[BotChatSession]:
        ChatManager._validate_platform(platform)

        try:
            streams = [
                stream
                for stream in _chat_manager.sessions.values()
                if ChatManager._match_platform(stream, platform)
                and (is_group_session is None or stream.is_group_session == is_group_session)
            ]
            return streams
        except Exception as e:
            logger.error(f"[ChatService] 获取聊天流失败: {e}")
            return []

    @staticmethod
    def _find_stream(
        predicate: Callable[[BotChatSession], bool],
        platform: Optional[str] | SpecialTypes = "qq",
    ) -> Optional[BotChatSession]:
        for stream in ChatManager._get_streams(platform=platform):
            if predicate(stream):
                return stream
        return None

    @staticmethod
    def get_all_streams(platform: Optional[str] | SpecialTypes = "qq") -> List[BotChatSession]:
        streams = ChatManager._get_streams(platform=platform)
        logger.debug(f"[ChatService] 获取到 {len(streams)} 个 {platform} 平台的聊天流")
        return streams

    @staticmethod
    def get_group_streams(platform: Optional[str] | SpecialTypes = "qq") -> List[BotChatSession]:
        streams = ChatManager._get_streams(platform=platform, is_group_session=True)
        logger.debug(f"[ChatService] 获取到 {len(streams)} 个 {platform} 平台的群聊流")
        return streams

    @staticmethod
    def get_private_streams(platform: Optional[str] | SpecialTypes = "qq") -> List[BotChatSession]:
        streams = ChatManager._get_streams(platform=platform, is_group_session=False)
        logger.debug(f"[ChatService] 获取到 {len(streams)} 个 {platform} 平台的私聊流")
        return streams

    @staticmethod
    def get_group_stream_by_group_id(
        group_id: str, platform: Optional[str] | SpecialTypes = "qq"
    ) -> Optional[BotChatSession]:  # sourcery skip: remove-unnecessary-cast
        if not isinstance(group_id, str):
            raise TypeError("group_id 必须是字符串类型")
        ChatManager._validate_platform(platform)
        if not group_id:
            raise ValueError("group_id 不能为空")
        try:
            stream = ChatManager._find_stream(
                lambda item: item.is_group_session and str(item.group_id) == str(group_id),
                platform=platform,
            )
            if stream is not None:
                logger.debug(f"[ChatService] 找到群ID {group_id} 的聊天流")
                return stream
            logger.warning(f"[ChatService] 未找到群ID {group_id} 的聊天流")
        except Exception as e:
            logger.error(f"[ChatService] 查找群聊流失败: {e}")
        return None

    @staticmethod
    def get_private_stream_by_user_id(
        user_id: str, platform: Optional[str] | SpecialTypes = "qq"
    ) -> Optional[BotChatSession]:  # sourcery skip: remove-unnecessary-cast
        if not isinstance(user_id, str):
            raise TypeError("user_id 必须是字符串类型")
        ChatManager._validate_platform(platform)
        if not user_id:
            raise ValueError("user_id 不能为空")
        try:
            stream = ChatManager._find_stream(
                lambda item: (not item.is_group_session) and str(item.user_id) == str(user_id),
                platform=platform,
            )
            if stream is not None:
                logger.debug(f"[ChatService] 找到用户ID {user_id} 的私聊流")
                return stream
            logger.warning(f"[ChatService] 未找到用户ID {user_id} 的私聊流")
        except Exception as e:
            logger.error(f"[ChatService] 查找私聊流失败: {e}")
        return None

    @staticmethod
    def get_stream_type(chat_stream: BotChatSession) -> str:
        if not isinstance(chat_stream, BotChatSession):
            raise TypeError("chat_stream 必须是 BotChatSession 类型")
        if not chat_stream:
            raise ValueError("chat_stream 不能为 None")

        return "group" if chat_stream.is_group_session else "private"

    @staticmethod
    def get_stream_info(chat_stream: BotChatSession) -> Dict[str, Any]:
        if not chat_stream:
            raise ValueError("chat_stream 不能为 None")
        if not isinstance(chat_stream, BotChatSession):
            raise TypeError("chat_stream 必须是 BotChatSession 类型")

        try:
            info: Dict[str, Any] = {
                "session_id": chat_stream.session_id,
                "platform": chat_stream.platform,
                "type": ChatManager.get_stream_type(chat_stream),
            }

            if chat_stream.is_group_session:
                info["group_id"] = chat_stream.group_id
                if (
                    chat_stream.context
                    and chat_stream.context.message
                    and chat_stream.context.message.message_info.group_info
                ):
                    info["group_name"] = chat_stream.context.message.message_info.group_info.group_name or "未知群聊"
                else:
                    info["group_name"] = "未知群聊"
            else:
                info["user_id"] = chat_stream.user_id
                if (
                    chat_stream.context
                    and chat_stream.context.message
                    and chat_stream.context.message.message_info.user_info
                ):
                    info["user_name"] = chat_stream.context.message.message_info.user_info.user_nickname
                else:
                    info["user_name"] = "未知用户"

            return info
        except Exception as e:
            logger.error(f"[ChatService] 获取聊天流信息失败: {e}")
            return {}
