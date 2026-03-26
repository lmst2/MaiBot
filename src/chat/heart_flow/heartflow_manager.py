from typing import Dict

import traceback

from src.chat.message_receive.chat_manager import chat_manager
from src.common.logger import get_logger
from src.maisaka.runtime import MaisakaHeartFlowChatting

logger = get_logger("heartflow")


class HeartflowManager:
    """主心流协调器。

    当前群聊统一使用 Maisaka runtime 作为消息核心循环实现。
    """

    def __init__(self) -> None:
        """初始化心流聊天实例缓存。"""
        self.heartflow_chat_list: Dict[str, MaisakaHeartFlowChatting] = {}

    async def get_or_create_heartflow_chat(self, session_id: str) -> MaisakaHeartFlowChatting:
        """获取或创建群聊心流实例。

        Args:
            session_id: 聊天会话 ID。

        Returns:
            MaisakaHeartFlowChatting: 当前会话绑定的 Maisaka runtime。
        """
        try:
            if chat := self.heartflow_chat_list.get(session_id):
                return chat
            chat_session = chat_manager.get_session_by_session_id(session_id)
            if not chat_session:
                raise ValueError(f"未找到 session_id={session_id} 的聊天流")
            new_chat = MaisakaHeartFlowChatting(session_id=session_id)
            await new_chat.start()
            self.heartflow_chat_list[session_id] = new_chat
            return new_chat
        except Exception as e:
            logger.error(f"创建心流聊天 {session_id} 失败: {e}", exc_info=True)
            traceback.print_exc()
            raise e

    def adjust_talk_frequency(self, session_id: str, frequency: float) -> None:
        """调整指定聊天流的说话频率。

        Args:
            session_id: 聊天会话 ID。
            frequency: 目标频率系数。
        """
        chat = self.heartflow_chat_list.get(session_id)
        if chat:
            chat.adjust_talk_frequency(frequency)
            logger.info(f"已调整聊天 {session_id} 的说话频率为 {frequency}")
        else:
            logger.warning(f"无法调整频率，未找到 session_id={session_id} 的聊天流")


heartflow_manager = HeartflowManager()
