from enum import Enum

import hashlib
import time


class TimestampMode(Enum):
    NORMAL = "%Y-%m-%d %H:%M:%S"
    """标准格式，例如 2024-01-01 12:00:00"""
    NORMAL_NO_YMD = "%H:%M:%S"
    """仅显示时间不显示年月日，例如 12:00:00"""
    RELATIVE = "relative"
    """相对时间，例如 5分钟前、2小时前等"""


def number_to_short_id(original_id: int, salt: str, length: int = 6) -> str:
    """
    将数字编号转换为短ID（不可逆）

    :param original_id: 原始数字
    :param length: 想要生成的短ID长度 (建议 4-8)
    :return: 短ID字符串
    """
    # 1. 加盐，避免简单的哈希冲突和猜测
    data = f"{original_id}{salt}".encode("utf-8")

    # 2. 计算 SHA-256 哈希
    hash_digest = hashlib.sha256(data).digest()

    # 3. 取前几个字节转换为整数
    # 为了达到需要的长度，我们可能需要取更多的字节
    num_bytes_needed = max(4, length)  # 保证足够的熵
    hash_int = int.from_bytes(hash_digest[:num_bytes_needed], byteorder="big")

    # 4. 使用 Base62 字符集编码
    characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    base = len(characters)

    short_id = ""
    temp_num = hash_int

    # 生成指定长度的ID
    for _ in range(length):
        short_id = characters[temp_num % base] + short_id
        temp_num //= base

    return short_id


def translate_timestamp_to_human_readable(timestamp: float, mode: TimestampMode) -> str:
    """将时间戳按照指定模式转换为人类可读的格式

    Args:
        timestamp (float): 需要转换的时间戳
        mode (TimestampMode): 时间戳转换模式，支持NORMAL、NORMAL_NO_YMD和RELATIVE三种模式
    Returns:
        str: 转换后的时间字符串
    """
    if mode in [TimestampMode.NORMAL, TimestampMode.NORMAL_NO_YMD]:
        return time.strftime(mode.value, time.localtime(timestamp))
    elif mode == TimestampMode.RELATIVE:
        time_diff = time.time() - timestamp

        if time_diff < 20:
            return "刚刚"
        elif time_diff < 60:
            return f"{int(time_diff)}秒前"
        elif time_diff < 3600:
            return f"{int(time_diff // 60)}分钟前"
        elif time_diff < 86400:
            return f"{int(time_diff // 3600)}小时前"
        elif time_diff < 2592000:
            return f"{int(time_diff // 86400)}天前"
        else:
            return time.strftime(TimestampMode.NORMAL.value, time.localtime(timestamp))
    else:
        raise ValueError(f"不支持的时间戳转换模式: {mode}")
