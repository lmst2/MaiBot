"""Maisaka 表情工具内置能力。"""

from dataclasses import dataclass, field
from typing import Sequence

import random

from src.common.data_models.image_data_model import MaiEmoji
from src.common.data_models.llm_service_data_models import LLMGenerationOptions
from src.common.logger import get_logger
from src.common.utils.utils_image import ImageUtils
from src.services import send_service

from .emoji_manager import emoji_manager, emoji_manager_emotion_judge_llm

logger = get_logger("emoji_maisaka_tool")


@dataclass(slots=True)
class MaisakaEmojiSendResult:
    """Maisaka 表情发送结果。"""

    success: bool
    message: str
    emoji_base64: str = ""
    description: str = ""
    emotions: list[str] = field(default_factory=list)
    requested_emotion: str = ""
    matched_emotion: str = ""


def _normalize_emotions(emoji: MaiEmoji) -> list[str]:
    """提取并清洗单个表情的情绪标签。"""

    return [str(item).strip() for item in emoji.emotion if str(item).strip()]


def _build_recent_context_text(context_texts: Sequence[str], max_items: int = 5) -> str:
    """构建供情绪判断使用的最近上下文文本。"""

    normalized_items = [str(item).strip() for item in context_texts if str(item).strip()]
    if not normalized_items:
        return ""
    return "\n".join(normalized_items[-max_items:])


async def _select_emoji_with_llm(
    *,
    sampled_emojis: Sequence[MaiEmoji],
    reasoning: str,
    context_text: str,
) -> tuple[MaiEmoji, str]:
    """让模型在采样表情中选择更合适的情绪标签。"""

    emotion_map: dict[str, list[MaiEmoji]] = {}
    for emoji in sampled_emojis:
        for emotion in _normalize_emotions(emoji):
            emotion_map.setdefault(emotion, []).append(emoji)

    available_emotions = list(emotion_map.keys())
    if not available_emotions:
        return random.choice(list(sampled_emojis)), ""

    prompt = (
        "你正在为聊天场景选择一个最合适的表情包情绪标签。\n"
        f"发送原因：{reasoning or '辅助表达当前语气和情绪'}\n"
        f"最近聊天记录：\n{context_text or '（暂无额外上下文）'}\n\n"
        "可选情绪标签如下：\n"
        f"{chr(10).join(available_emotions)}\n\n"
        "请只返回一个最匹配的情绪标签，不要解释。"
    )

    try:
        llm_result = await emoji_manager_emotion_judge_llm.generate_response(
            prompt,
            options=LLMGenerationOptions(temperature=0.3, max_tokens=60),
        )
        chosen_emotion = (llm_result.response or "").strip().strip("\"'")
    except Exception as exc:
        logger.warning(f"使用 LLM 选择表情情绪失败，将回退为随机选择: {exc}")
        chosen_emotion = ""

    if chosen_emotion and chosen_emotion in emotion_map:
        return random.choice(emotion_map[chosen_emotion]), chosen_emotion
    return random.choice(list(sampled_emojis)), ""


async def select_emoji_for_maisaka(
    *,
    requested_emotion: str = "",
    reasoning: str = "",
    context_texts: Sequence[str] | None = None,
    sample_size: int = 30,
) -> tuple[MaiEmoji | None, str]:
    """为 Maisaka 选择一个合适的表情。"""

    available_emojis = list(emoji_manager.emojis)
    if not available_emojis:
        return None, ""

    normalized_requested_emotion = requested_emotion.strip()
    if normalized_requested_emotion:
        matched_emojis = [
            emoji
            for emoji in available_emojis
            if normalized_requested_emotion.lower() in (emotion.lower() for emotion in _normalize_emotions(emoji))
        ]
        if matched_emojis:
            return random.choice(matched_emojis), normalized_requested_emotion

    sampled_emojis = random.sample(
        available_emojis,
        min(max(sample_size, 1), len(available_emojis)),
    )
    context_text = _build_recent_context_text(context_texts or [])
    return await _select_emoji_with_llm(
        sampled_emojis=sampled_emojis,
        reasoning=reasoning,
        context_text=context_text,
    )


async def send_emoji_for_maisaka(
    *,
    stream_id: str,
    requested_emotion: str = "",
    reasoning: str = "",
    context_texts: Sequence[str] | None = None,
) -> MaisakaEmojiSendResult:
    """为 Maisaka 选择并发送一个表情。"""

    selected_emoji, matched_emotion = await select_emoji_for_maisaka(
        requested_emotion=requested_emotion,
        reasoning=reasoning,
        context_texts=context_texts,
    )
    if selected_emoji is None:
        return MaisakaEmojiSendResult(
            success=False,
            message="当前表情包库中没有可用表情。",
            requested_emotion=requested_emotion.strip(),
        )

    try:
        emoji_base64 = ImageUtils.image_path_to_base64(str(selected_emoji.full_path))
        if not emoji_base64:
            raise ValueError("表情图片转换为 base64 失败")
    except Exception as exc:
        return MaisakaEmojiSendResult(
            success=False,
            message=f"发送表情包失败：{exc}",
            description=selected_emoji.description.strip(),
            emotions=_normalize_emotions(selected_emoji),
            requested_emotion=requested_emotion.strip(),
            matched_emotion=matched_emotion,
        )

    try:
        sent = await send_service.emoji_to_stream(
            emoji_base64=emoji_base64,
            stream_id=stream_id,
            storage_message=True,
            set_reply=False,
            reply_message=None,
        )
    except Exception as exc:
        return MaisakaEmojiSendResult(
            success=False,
            message=f"发送表情包时发生异常：{exc}",
            description=selected_emoji.description.strip(),
            emotions=_normalize_emotions(selected_emoji),
            requested_emotion=requested_emotion.strip(),
            matched_emotion=matched_emotion,
        )

    description = selected_emoji.description.strip()
    emotions = _normalize_emotions(selected_emoji)
    if not sent:
        return MaisakaEmojiSendResult(
            success=False,
            message="发送表情包失败。",
            description=description,
            emotions=emotions,
            requested_emotion=requested_emotion.strip(),
            matched_emotion=matched_emotion,
        )

    emoji_manager.update_emoji_usage(selected_emoji)
    success_message = (
        f"已发送表情包：{description}（情绪：{', '.join(emotions)}）"
        if emotions
        else f"已发送表情包：{description}"
    )
    return MaisakaEmojiSendResult(
        success=True,
        message=success_message,
        emoji_base64=emoji_base64,
        description=description,
        emotions=emotions,
        requested_emotion=requested_emotion.strip(),
        matched_emotion=matched_emotion,
    )
