"""
Maisaka knowledge retrieval and learning helpers.
"""

from typing import Any, Dict, List

import asyncio
import json

from src.chat.message_receive.message import SessionMessage
from src.chat.utils.utils import is_bot_self
from src.common.data_models.llm_service_data_models import LLMGenerationOptions
from src.common.logger import get_logger
from src.services.llm_service import LLMServiceClient

from src.know_u.knowledge_store import KNOWLEDGE_CATEGORIES, get_knowledge_store
from src.maisaka.message_adapter import get_message_role, get_message_text, parse_speaker_content

logger = get_logger("maisaka_knowledge")

NO_RESULT_KEYWORDS = [
    "无",
    "没有",
    "不适用",
    "无需",
    "无相关",
]


def extract_category_ids_from_result(result: str) -> List[str]:
    """Extract valid category ids from an LLM result string."""
    if not result:
        return []

    normalized = result.strip()
    if not normalized:
        return []

    lowered = normalized.lower()
    if any(keyword in lowered for keyword in ["none", "no relevant", "no_need", "no need"]):
        return []
    if any(keyword in normalized for keyword in NO_RESULT_KEYWORDS):
        return []

    category_ids: List[str] = []
    for part in normalized.replace(",", " ").replace("，", " ").replace("\n", " ").split():
        candidate = part.strip()
        if candidate in KNOWLEDGE_CATEGORIES and candidate not in category_ids:
            category_ids.append(candidate)

    return category_ids


async def retrieve_relevant_knowledge(
    knowledge_analyzer: Any,
    chat_history: List[SessionMessage],
) -> str:
    """Retrieve formatted knowledge snippets relevant to the current chat history."""
    store = get_knowledge_store()
    categories_summary = store.get_categories_summary()

    try:
        category_ids = await knowledge_analyzer.analyze_knowledge_need(chat_history, categories_summary)
        if not category_ids:
            return ""
        return store.get_formatted_knowledge(category_ids)
    except Exception:
        logger.exception("Failed to retrieve relevant knowledge")
        return ""


class KnowledgeLearner:
    """
    从最近对话中提取用户画像类知识并写入知识库。
    """

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._store = get_knowledge_store()
        self._llm = LLMServiceClient(task_name="utils", request_type="maisaka.knowledge.learn")
        self._learning_lock = asyncio.Lock()
        self._messages_cache: List[SessionMessage] = []

    def add_messages(self, messages: List[SessionMessage]) -> None:
        """缓存待学习的消息。"""
        self._messages_cache.extend(messages)

    def get_cache_size(self) -> int:
        """获取缓存消息数量。"""
        return len(self._messages_cache)

    async def learn(self) -> int:
        """
        从缓存消息中提取知识并落库。

        Returns:
            新增入库的知识条数
        """
        if not self._messages_cache:
            return 0

        async with self._learning_lock:
            chat_excerpt = self._build_chat_excerpt()
            if not chat_excerpt:
                return 0

            prompt = self._build_learning_prompt(chat_excerpt)
            try:
                result = await self._llm.generate_response(
                    prompt=prompt,
                    options=LLMGenerationOptions(
                        temperature=0.1,
                        max_tokens=512,
                    ),
                )
            except Exception:
                logger.exception("Knowledge learning model call failed")
                return 0

            knowledge_items = self._parse_learning_result(result.response or "")
            if not knowledge_items:
                logger.debug("Knowledge learning finished without extracted entries")
                return 0

            added_count = 0
            for item in knowledge_items:
                category_id = str(item.get("category_id", "")).strip()
                content = str(item.get("content", "")).strip()
                if not category_id or not content:
                    continue

                if self._store.add_knowledge(
                    category_id=category_id,
                    content=content,
                    metadata={
                        "session_id": self._session_id,
                        "source": "maisaka_learning",
                    },
                ):
                    added_count += 1

            if added_count > 0:
                logger.info(
                    f"Maisaka knowledge learning finished: session_id={self._session_id} added={added_count}"
                )
            else:
                logger.debug(
                    f"Maisaka knowledge learning finished without new entries: session_id={self._session_id}"
                )

            return added_count

    def _build_chat_excerpt(self) -> str:
        """
        构建适合画像提取的对话片段，只保留用户可见文本。
        """
        lines: List[str] = []
        for message in self._messages_cache[-30:]:
            if get_message_role(message) == "assistant":
                continue
            if get_message_role(message) == "tool":
                continue
            if is_bot_self(message.platform, message.message_info.user_info.user_id):
                continue

            raw_text = get_message_text(message).strip()
            if not raw_text:
                continue

            speaker_name, body = parse_speaker_content(raw_text)
            visible_text = (body or raw_text).strip()
            if not visible_text:
                continue

            speaker = speaker_name or message.message_info.user_info.user_nickname or "用户"
            lines.append(f"{speaker}: {visible_text}")

        return "\n".join(lines)

    def _build_learning_prompt(self, chat_excerpt: str) -> str:
        """构建知识提取提示词。"""
        categories_text = "\n".join(
            f"{category_id}. {category_name}" for category_id, category_name in KNOWLEDGE_CATEGORIES.items()
        )
        return (
            "你是一个用户画像知识提取器，需要从聊天记录里提取稳定、可复用的用户事实。\n"
            "只提取用户明确表达或高置信度可归纳的信息，不要猜测，不要提取一次性情绪，不要重复表述。\n"
            "如果没有可提取内容，返回空数组 []。\n"
            "输出必须是 JSON 数组，每项格式为 "
            '{"category_id":"分类编号","content":"简洁中文陈述"}。\n'
            "分类如下：\n"
            f"{categories_text}\n\n"
            "聊天记录：\n"
            f"{chat_excerpt}"
        )

    def _parse_learning_result(self, result: str) -> List[Dict[str, str]]:
        """解析模型返回的知识条目。"""
        normalized = result.strip()
        if not normalized:
            return []

        if "```" in normalized:
            normalized = normalized.replace("```json", "").replace("```JSON", "").replace("```", "").strip()

        try:
            parsed = json.loads(normalized)
        except json.JSONDecodeError:
            logger.warning("Knowledge learning result is not valid JSON")
            return []

        if not isinstance(parsed, list):
            return []

        normalized_items: List[Dict[str, str]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for item in parsed:
            if not isinstance(item, dict):
                continue

            category_id = str(item.get("category_id", "")).strip()
            content = " ".join(str(item.get("content", "")).strip().split())
            if category_id not in KNOWLEDGE_CATEGORIES:
                continue
            if not content:
                continue

            pair = (category_id, content)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            normalized_items.append(
                {
                    "category_id": category_id,
                    "content": content,
                }
            )

        return normalized_items
