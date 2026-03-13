from typing import Dict

import traceback

from src.common.logger import get_logger
from src.chat.message_receive.chat_manager import chat_manager
from src.chat.heart_flow.heartFC_chat import HeartFChatting
# from src.chat.brain_chat.brain_chat import BrainChatting

logger = get_logger("heartflow")


# TODO: 恢复PFC，现在暂时禁用
class HeartflowManager:
    """主心流协调器，负责初始化并协调聊天，控制聊天属性"""

    def __init__(self):
        # self.heartflow_chat_list: Dict[str, HeartFChatting | BrainChatting] = {}
        self.heartflow_chat_list: Dict[str, HeartFChatting] = {}

    async def get_or_create_heartflow_chat(self, session_id: str):  # -> Optional[HeartFChatting | BrainChatting]:
        """获取或创建一个新的HeartFChatting实例"""
        try:
            if chat := self.heartflow_chat_list.get(session_id):
                return chat
            chat_session = chat_manager.get_session_by_session_id(session_id)
            if not chat_session:
                raise ValueError(f"未找到 session_id={session_id} 的聊天流")
            # new_chat = (
            #     HeartFChatting(session_id=session_id) if chat_session.group_id else BrainChatting(session_id=session_id)
            # )
            new_chat = HeartFChatting(session_id=session_id)
            await new_chat.start()
            self.heartflow_chat_list[session_id] = new_chat
            return new_chat
        except Exception as e:
            logger.error(f"创建心流聊天 {session_id} 失败: {e}", exc_info=True)
            traceback.print_exc()
            raise e

    def adjust_talk_frequency(self, session_id: str, frequency: float):
        """调整指定聊天流的说话频率"""
        chat = self.heartflow_chat_list.get(session_id)
        if chat and isinstance(chat, HeartFChatting):
            chat.adjust_talk_frequency(frequency)
            logger.info(f"已调整聊天 {session_id} 的说话频率为 {frequency}")
        else:
            logger.warning(f"无法调整频率，未找到 session_id={session_id} 的聊天流")


heartflow_manager = HeartflowManager()
