"""
MaiSaka - Timing 模块（含自我反思功能）
构建对话时间戳信息，供 Timing 分析模块使用。
该模块同时负责分析对话的时间维度和进行自我反思分析。
"""

from datetime import datetime
from typing import Optional


def build_timing_info(
    chat_start_time: Optional[datetime],
    last_user_input_time: Optional[datetime],
    last_assistant_response_time: Optional[datetime],
    user_input_times: list[datetime],
) -> str:
    """
    构建当前时间戳信息文本，供 Timing 模块分析。

    Args:
        chat_start_time:             对话开始时间
        last_user_input_time:        用户上次输入时间
        last_assistant_response_time: 助手上次回复时间
        user_input_times:            所有用户输入时间戳列表
    """
    now = datetime.now()
    parts: list[str] = []

    parts.append(f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")

    if chat_start_time:
        elapsed = now - chat_start_time
        minutes, seconds = divmod(int(elapsed.total_seconds()), 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            parts.append(f"对话已持续: {hours}小时{minutes}分{seconds}秒")
        elif minutes > 0:
            parts.append(f"对话已持续: {minutes}分{seconds}秒")
        else:
            parts.append(f"对话已持续: {seconds}秒")

    if last_user_input_time:
        since_user = now - last_user_input_time
        parts.append(f"距用户上次输入: {int(since_user.total_seconds())}秒")

    if last_assistant_response_time:
        since_assistant = now - last_assistant_response_time
        parts.append(f"距助手上次回复: {int(since_assistant.total_seconds())}秒")

    if len(user_input_times) >= 2:
        intervals = [
            (user_input_times[i] - user_input_times[i - 1]).total_seconds() for i in range(1, len(user_input_times))
        ]
        avg_interval = sum(intervals) / len(intervals)
        parts.append(f"用户平均回复间隔: {int(avg_interval)}秒")
        parts.append(f"用户总共发言次数: {len(user_input_times)}")

    # 时段判断
    hour = now.hour
    if 0 <= hour < 6:
        parts.append("当前时段: 深夜/凌晨")
    elif 6 <= hour < 9:
        parts.append("当前时段: 早晨")
    elif 9 <= hour < 12:
        parts.append("当前时段: 上午")
    elif 12 <= hour < 14:
        parts.append("当前时段: 中午")
    elif 14 <= hour < 18:
        parts.append("当前时段: 下午")
    elif 18 <= hour < 22:
        parts.append("当前时段: 晚上")
    else:
        parts.append("当前时段: 深夜")

    return "\n".join(parts)
