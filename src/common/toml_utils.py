"""
TOML文件工具函数 - 保留格式和注释
"""

import os
import tomlkit
from typing import Any


def save_toml_with_format(data: dict[str, Any], file_path: str) -> None:
    """
    保存TOML数据到文件，保留现有格式（如果文件存在）

    Args:
        data: 要保存的数据字典
        file_path: 文件路径
    """
    # 如果文件不存在，直接创建
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            tomlkit.dump(data, f)
        return

    # 如果文件存在，尝试读取现有文件以保留格式
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            existing_doc = tomlkit.load(f)
    except Exception:
        # 如果读取失败，直接覆盖
        with open(file_path, "w", encoding="utf-8") as f:
            tomlkit.dump(data, f)
        return

    # 递归更新，保留现有格式
    _merge_toml_preserving_format(existing_doc, data)

    # 保存
    with open(file_path, "w", encoding="utf-8") as f:
        tomlkit.dump(existing_doc, f)


def _merge_toml_preserving_format(target: dict[str, Any], source: dict[str, Any]) -> None:
    """
    递归合并source到target，保留target中的格式和注释

    Args:
        target: 目标文档（保留格式）
        source: 源数据（新数据）
    """
    for key, value in source.items():
        if key in target:
            # 如果两个都是字典且都是表格，递归合并
            if isinstance(value, dict) and isinstance(target[key], dict):
                if hasattr(target[key], "items"):  # 确实是字典/表格
                    _merge_toml_preserving_format(target[key], value)
                else:
                    target[key] = value
            else:
                # 其他情况直接替换
                target[key] = value
        else:
            # 新键直接添加
            target[key] = value


def _update_toml_doc(target: dict[str, Any], source: dict[str, Any]) -> None:
    """
    更新TOML文档中的字段，保留现有的格式和注释

    这是一个递归函数，用于在部分更新配置时保留现有的格式和注释。

    Args:
        target: 目标表格（会被修改）
        source: 源数据（新数据）
    """
    for key, value in source.items():
        if key in target:
            # 如果两个都是字典，递归更新
            if isinstance(value, dict) and isinstance(target[key], dict):
                if hasattr(target[key], "items"):  # 确实是表格
                    _update_toml_doc(target[key], value)
                else:
                    target[key] = value
            else:
                # 直接更新值，保留注释
                target[key] = value
        else:
            # 新键直接添加
            target[key] = value
