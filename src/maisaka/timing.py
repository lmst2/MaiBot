"""
MaiSaka timing helpers.
"""

from datetime import datetime
from typing import Optional


def _format_duration(total_seconds: int) -> str:
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _get_time_period_label(hour: int) -> str:
    if 0 <= hour < 6:
        return "late_night"
    if 6 <= hour < 9:
        return "morning"
    if 9 <= hour < 12:
        return "late_morning"
    if 12 <= hour < 14:
        return "noon"
    if 14 <= hour < 18:
        return "afternoon"
    if 18 <= hour < 22:
        return "evening"
    return "night"


def build_timing_info(
    chat_start_time: Optional[datetime],
    last_user_input_time: Optional[datetime],
    last_assistant_response_time: Optional[datetime],
    user_input_times: list[datetime],
) -> str:
    """Build readable timing context for the timing analysis prompt."""
    now = datetime.now()
    parts: list[str] = [f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}"]

    if chat_start_time:
        elapsed_seconds = int((now - chat_start_time).total_seconds())
        parts.append(f"Conversation duration: {_format_duration(elapsed_seconds)}")

    if last_user_input_time:
        since_user_seconds = int((now - last_user_input_time).total_seconds())
        parts.append(f"Seconds since last user input: {since_user_seconds}")

    if last_assistant_response_time:
        since_assistant_seconds = int((now - last_assistant_response_time).total_seconds())
        parts.append(f"Seconds since last Maisaka reply: {since_assistant_seconds}")

    if len(user_input_times) >= 2:
        intervals = [
            int((user_input_times[index] - user_input_times[index - 1]).total_seconds())
            for index in range(1, len(user_input_times))
        ]
        average_interval = sum(intervals) / len(intervals)
        parts.append(f"Average user input interval: {int(average_interval)}s")
        parts.append(f"Total user input count: {len(user_input_times)}")

    parts.append(f"Current time period: {_get_time_period_label(now.hour)}")
    return "\n".join(parts)
