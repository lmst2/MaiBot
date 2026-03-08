from json_repair import repair_json
from typing import List, Tuple

import re
import json

from src.common.logger import get_logger

logger = get_logger("learner_utils")


def fix_chinese_quotes_in_json(text):
    """使用状态机修复 JSON 字符串值中的中文引号"""
    result = []
    i = 0
    in_string = False
    escape_next = False

    while i < len(text):
        char = text[i]
        if escape_next:
            # 当前字符是转义字符后的字符，直接添加
            result.append(char)
            escape_next = False
            i += 1
            continue
        if char == "\\":
            # 转义字符
            result.append(char)
            escape_next = True
            i += 1
            continue
        if char == '"' and not escape_next:
            # 遇到英文引号，切换字符串状态
            in_string = not in_string
            result.append(char)
            i += 1
            continue
        if in_string and char in ["“", "”"]:
            result.append('\\"')
        else:
            result.append(char)
        i += 1

    return "".join(result)


def parse_expression_response(response: str) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str]]]:
    """
    解析 LLM 返回的表达风格总结和黑话 JSON，提取两个列表。

    期望的 JSON 结构：
    [
        {"situation": "AAAAA", "style": "BBBBB", "source_id": "3"},  // 表达方式
        {"content": "词条", "source_id": "12"},  // 黑话
        ...
    ]

    Returns:
        Tuple[List[Tuple[str, str, str]], List[Tuple[str, str]]]:
            第一个列表是表达方式 (situation, style, source_id)
            第二个列表是黑话 (content, source_id)
    """
    if not response:
        return [], []

    raw = response.strip()

    if match := re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL):
        raw = match[1].strip()
    else:
        # 去掉可能存在的通用 ``` 包裹
        raw = re.sub(r"^```\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
        raw = raw.strip()

    parsed = None
    expressions: List[Tuple[str, str, str]] = []  # (situation, style, source_id)
    jargon_entries: List[Tuple[str, str]] = []  # (content, source_id)

    try:
        # 优先尝试直接解析
        if raw.startswith("[") and raw.endswith("]"):
            parsed = json.loads(raw)
        else:
            repaired = repair_json(raw)
            parsed = json.loads(repaired) if isinstance(repaired, str) else repaired
    except Exception as parse_error:
        # 如果解析失败，尝试修复中文引号问题
        # 使用状态机方法，在 JSON 字符串值内部将中文引号替换为转义的英文引号
        try:
            fixed_raw = fix_chinese_quotes_in_json(raw)

            # 再次尝试解析
            if fixed_raw.startswith("[") and fixed_raw.endswith("]"):
                parsed = json.loads(fixed_raw)
            else:
                repaired = repair_json(fixed_raw)
                parsed = json.loads(repaired) if isinstance(repaired, str) else repaired
        except Exception as fix_error:
            logger.error(f"解析表达风格 JSON 失败，初始错误: {type(parse_error).__name__}: {str(parse_error)}")
            logger.error(f"修复中文引号后仍失败，错误: {type(fix_error).__name__}: {str(fix_error)}")
            logger.error(f"解析表达风格 JSON 失败，原始响应：{response}")
            logger.error(f"处理后的 JSON 字符串（前500字符）：{raw[:500]}")
            return [], []

    if isinstance(parsed, dict):
        parsed_list = [parsed]
    elif isinstance(parsed, list):
        parsed_list = parsed
    else:
        logger.error(f"表达风格解析结果类型异常: {type(parsed)}, 内容: {parsed}")
        return [], []

    for item in parsed_list:
        if not isinstance(item, dict):
            continue

        # 检查是否是表达方式条目（有 situation 和 style）
        situation = str(item.get("situation", "")).strip()
        style = str(item.get("style", "")).strip()
        source_id = str(item.get("source_id", "")).strip()

        if situation and style and source_id:
            # 表达方式条目
            expressions.append((situation, style, source_id))
        elif item.get("content"):
            # 黑话条目（有 content 字段）
            content = str(item.get("content", "")).strip()
            source_id = str(item.get("source_id", "")).strip()
            if content and source_id:
                jargon_entries.append((content, source_id))

    return expressions, jargon_entries
