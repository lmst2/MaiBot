from dataclasses import dataclass
from typing import Optional, Dict, Any

import time

from src.config.config import global_config
from src.common.logger import get_logger
from src.chat.message_receive.chat_manager import chat_manager as _chat_manager
from src.services import send_service as send_api

from src.common.message_repository import count_messages

logger = get_logger(__name__)


@dataclass
class CyclePlanInfo: ...


@dataclass
class CycleActionInfo: ...


class CycleDetail:
    """循环信息记录类"""

    def __init__(self, cycle_id: int):
        self.cycle_id = cycle_id
        self.thinking_id = ""
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.timers: Dict[str, float] = {}


    def set_loop_info(self, loop_info: Dict[str, Any]):
        """设置循环信息"""
        self.loop_plan_info = loop_info["loop_plan_info"]
        self.loop_action_info = loop_info["loop_action_info"]


def get_recent_message_stats(minutes: float = 30, chat_id: Optional[str] = None) -> dict:
    """
    Args:
        minutes (float): 检索的分钟数，默认30分钟
        chat_id (str, optional): 指定的chat_id，仅统计该chat下的消息。为None时统计全部。
    Returns:
        dict: {"bot_reply_count": int, "total_message_count": int}
    """

    now = time.time()
    start_time = now - minutes * 60
    bot_id = global_config.bot.qq_account

    filter_base: Dict[str, Any] = {"time": {"$gte": start_time}}
    if chat_id is not None:
        filter_base["chat_id"] = chat_id

    # 总消息数
    total_message_count = count_messages(filter_base)
    # bot自身回复数
    bot_filter = filter_base.copy()
    bot_filter["user_id"] = bot_id
    bot_reply_count = count_messages(bot_filter)

    return {"bot_reply_count": bot_reply_count, "total_message_count": total_message_count}


async def send_typing():
    chat = await _chat_manager.get_or_create_session(
        platform="amaidesu_default",
        user_id="114514",
        group_id="114514",
    )

    await send_api.custom_to_stream(
        message_type="state", content="typing", stream_id=chat.session_id, storage_message=False
    )


async def stop_typing():
    chat = await _chat_manager.get_or_create_session(
        platform="amaidesu_default",
        user_id="114514",
        group_id="114514",
    )

    await send_api.custom_to_stream(
        message_type="state", content="stop_typing", stream_id=chat.session_id, storage_message=False
    )
