"""
MaiSaka knowledge retrieval helpers.
"""

from typing import List

from src.common.data_models.mai_message_data_model import MaiMessage

from .knowledge_store import KNOWLEDGE_CATEGORIES, get_knowledge_store

NO_RESULT_KEYWORDS = [
    "\u65e0",
    "\u6ca1\u6709",
    "\u4e0d\u9002\u7528",
    "\u65e0\u9700",
    "\u65e0\u76f8\u5173",
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
    for part in normalized.replace(",", " ").replace("\uff0c", " ").replace("\n", " ").split():
        candidate = part.strip()
        if candidate in KNOWLEDGE_CATEGORIES and candidate not in category_ids:
            category_ids.append(candidate)

    return category_ids


async def retrieve_relevant_knowledge(
    llm_service,
    chat_history: List[MaiMessage],
) -> str:
    """Retrieve formatted knowledge snippets relevant to the current chat history."""
    store = get_knowledge_store()
    categories_summary = store.get_categories_summary()

    try:
        category_ids = await llm_service.analyze_knowledge_need(chat_history, categories_summary)
        if not category_ids:
            return ""
        return store.get_formatted_knowledge(category_ids)
    except Exception:
        return ""
