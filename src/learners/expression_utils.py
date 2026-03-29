from json_repair import repair_json
from typing import Any, List, Optional, Tuple

import json
import re

from src.config.config import global_config
from src.common.data_models.llm_service_data_models import LLMGenerationOptions
from src.services.llm_service import LLMServiceClient
from src.prompt.prompt_manager import prompt_manager
from src.common.logger import get_logger

logger = get_logger("expression_utils")

judge_llm = LLMServiceClient(task_name="utils", request_type="expression_check")


def _normalize_repair_json_result(repaired_result: Any) -> str:
    """将 repair_json 的返回值规范化为 JSON 字符串。

    Args:
        repaired_result: `repair_json` 的返回值，可能是字符串或带附加信息的元组。

    Returns:
        str: 可供 `json.loads` 继续解析的 JSON 字符串。

    Raises:
        TypeError: 当返回值无法规范化为字符串时抛出。
    """
    if isinstance(repaired_result, str):
        return repaired_result
    if isinstance(repaired_result, tuple) and repaired_result:
        first_item = repaired_result[0]
        if isinstance(first_item, str):
            return first_item
        return json.dumps(first_item, ensure_ascii=False)
    raise TypeError(f"repair_json 返回了无法处理的结果类型: {type(repaired_result)}")


async def check_expression_suitability(situation: str, style: str) -> Tuple[bool, str, Optional[str]]:
    """
    执行单次LLM评估

    Args:
        situation: 情境
        style: 风格

    Returns:
        (suitable, reason, error) 元组，如果出错则 suitable 为 False，error 包含错误信息
    """
    # 构建评估提示词
    # 基础评估标准
    base_criteria = [
        "表达方式或言语风格是否与使用条件或使用情景匹配",
        "允许部分语法错误或口头化或缺省出现",
        "表达方式不能太过特指，需要具有泛用性",
        "一般不涉及具体的人名或名称",
    ]

    if custom_criteria := global_config.expression.expression_auto_check_custom_criteria:
        base_criteria.extend(custom_criteria)

    # 构建评估标准列表字符串
    criteria_list = "\n".join([f"{i + 1}. {criterion}" for i, criterion in enumerate(base_criteria)])

    prompt_template = prompt_manager.get_prompt("expression_evaluation")
    prompt_template.add_context("situation", situation)
    prompt_template.add_context("style", style)
    prompt_template.add_context("criteria_list", criteria_list)

    prompt = await prompt_manager.render_prompt(prompt_template)

    logger.info(f"正在评估表达方式: situation={situation}, style={style}")

    generation_result = await judge_llm.generate_response(
        prompt=prompt,
        options=LLMGenerationOptions(temperature=0.6, max_tokens=1024),
    )
    response = generation_result.response

    logger.debug(f"评估结果: {response}")

    try:
        evaluation = json.loads(response)
    except json.JSONDecodeError:
        try:
            response_repaired = _normalize_repair_json_result(repair_json(response))
            evaluation = json.loads(response_repaired)
        except Exception as e:
            raise ValueError(f"无法解析LLM响应为JSON: {response}") from e
    except Exception as e:
        return False, f"评估表达方式时发生错误: {e}", str(e)
    try:
        suitable = evaluation.get("suitable", False)
        reason = evaluation.get("reason", "未提供理由")
        logger.debug(f"评估结果: {'通过' if suitable else '不通过'}")
        return suitable, reason, None
    except Exception as e:
        return False, f"评估结果格式错误: {e}", str(e)


def fix_chinese_quotes_in_json(text: str) -> str:
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

    parsed = _try_parse(raw)
    if parsed is None:
        fixed = fix_chinese_quotes_in_json(raw)
        parsed = _try_parse(fixed)
    if parsed is None:
        logger.error(f"处理后的 JSON 字符串（前500字符）：{raw[:500]}")
        return [], []

    if isinstance(parsed, dict):
        parsed_list = [parsed]
    elif isinstance(parsed, list):
        parsed_list = parsed
    else:
        logger.error(f"表达风格解析结果类型异常: {type(parsed)}, 内容: {parsed}")
        return [], []

    expressions: List[Tuple[str, str, str]] = []  # (situation, style, source_id)
    jargon_entries: List[Tuple[str, str]] = []  # (content, source_id)

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
            continue
        content = str(item.get("content", "")).strip()
        if content and source_id:
            jargon_entries.append((content, source_id))

    return expressions, jargon_entries


def is_single_char_jargon(content: str) -> bool:
    """
    判断是否是单字黑话（单个汉字、英文或数字）

    Args:
        content: 词条内容

    Returns:
        bool: 如果是单字黑话返回True，否则返回False
    """
    if not content or len(content) != 1:
        return False

    char = content[0]
    # 判断是否是单个汉字、单个英文字母或单个数字
    return (
        "\u4e00" <= char <= "\u9fff"  # 汉字
        or "a" <= char <= "z"  # 小写字母
        or "A" <= char <= "Z"  # 大写字母
        or "0" <= char <= "9"  # 数字
    )


def _try_parse(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        try:
            repaired = _normalize_repair_json_result(repair_json(text))
            return json.loads(repaired)
        except Exception:
            return None
