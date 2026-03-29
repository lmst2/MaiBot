from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import json
import re

from sqlmodel import select

from src.chat.message_receive.message import SessionMessage
from src.common.database.database import get_db_session
from src.common.database.database_model import Expression, Jargon
from src.common.logger import get_logger
from src.config.config import global_config
from src.learners.jargon_explainer import search_jargon

from .message_adapter import get_message_text, parse_speaker_content

logger = get_logger("maisaka_reply_context")


@dataclass
class ReplyContextBuildResult:
    """Reply 前置上下文构建结果。"""

    expression_habits: str = ""
    jargon_explanation: str = ""
    selected_expression_ids: List[int] = field(default_factory=list)


@dataclass
class _ExpressionRecord:
    expression_id: Optional[int]
    situation: str
    style: str


@dataclass
class _JargonRecord:
    jargon_id: Optional[int]
    content: str
    meaning: str
    session_id_dict: str
    is_global: bool


class MaisakaReplyContextBuilder:
    """为 Maisaka reply 构建表达方式和黑话解释。"""

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id

    async def build(
        self,
        chat_history: List[SessionMessage],
        reply_message: Optional[SessionMessage],
        reply_reason: str,
        unknown_words: Optional[List[str]] = None,
    ) -> ReplyContextBuildResult:
        """构建 reply 前置上下文。"""
        expression_habits, selected_expression_ids = self._build_expression_habits(
            chat_history=chat_history,
            reply_message=reply_message,
            reply_reason=reply_reason,
        )
        jargon_explanation = self._build_jargon_explanation(
            chat_history=chat_history,
            reply_message=reply_message,
            unknown_words=unknown_words,
        )
        return ReplyContextBuildResult(
            expression_habits=expression_habits,
            jargon_explanation=jargon_explanation,
            selected_expression_ids=selected_expression_ids,
        )

    def _build_expression_habits(
        self,
        chat_history: List[SessionMessage],
        reply_message: Optional[SessionMessage],
        reply_reason: str,
    ) -> tuple[str, List[int]]:
        """查询并格式化适合当前会话的表达方式。"""
        del chat_history
        del reply_message
        del reply_reason

        expression_records = self._load_expression_records()
        if not expression_records:
            return "", []

        lines: List[str] = []
        selected_ids: List[int] = []
        for expression in expression_records:
            if expression.expression_id is not None:
                selected_ids.append(expression.expression_id)
            lines.append(f"- 当{expression.situation}时，可以自然地用{expression.style}这种表达习惯。")

        block = "【表达习惯参考】\n" + "\n".join(lines)
        logger.info(
            f"Built Maisaka expression habits: session_id={self._session_id} "
            f"count={len(selected_ids)} ids={selected_ids!r}"
        )
        return block, selected_ids

    def _load_expression_records(self) -> List[_ExpressionRecord]:
        """在 session 内提取表达方式的静态数据，避免 detached ORM 对象。"""
        with get_db_session(auto_commit=False) as session:
            query = select(Expression).where(Expression.rejected.is_(False))  # type: ignore[attr-defined]
            if global_config.expression.expression_checked_only:
                query = query.where(Expression.checked.is_(True))  # type: ignore[attr-defined]

            query = query.where(
                (Expression.session_id == self._session_id) | (Expression.session_id.is_(None))  # type: ignore[attr-defined]
            ).order_by(Expression.count.desc(), Expression.last_active_time.desc())  # type: ignore[attr-defined]

            expressions = session.exec(query.limit(5)).all()
            return [
                _ExpressionRecord(
                    expression_id=expression.id,
                    situation=expression.situation,
                    style=expression.style,
                )
                for expression in expressions
            ]

    def _build_jargon_explanation(
        self,
        chat_history: List[SessionMessage],
        reply_message: Optional[SessionMessage],
        unknown_words: Optional[List[str]],
    ) -> str:
        """查询并格式化黑话解释。"""
        if not global_config.expression.enable_jargon_explanation:
            return ""

        if global_config.expression.jargon_mode == "planner":
            return self._build_planner_jargon_explanation(unknown_words or [])

        return self._build_context_jargon_explanation(chat_history, reply_message)

    def _build_planner_jargon_explanation(self, unknown_words: List[str]) -> str:
        """基于 planner 传入的 unknown_words 构建黑话解释。"""
        normalized_words: List[str] = []
        seen_words: set[str] = set()
        for raw_word in unknown_words:
            word = str(raw_word or "").strip()
            if not word:
                continue
            lowered = word.lower()
            if lowered in seen_words:
                continue
            seen_words.add(lowered)
            normalized_words.append(word)

        if not normalized_words:
            return ""

        lines: List[str] = []
        seen_entries: set[str] = set()
        for word in normalized_words:
            matches = search_jargon(word, chat_id=self._session_id, limit=3, fuzzy=False)
            if not matches:
                matches = search_jargon(word, chat_id=self._session_id, limit=3, fuzzy=True)
            for match in matches:
                content = str(match.get("content") or "").strip()
                meaning = str(match.get("meaning") or "").strip()
                if not content or not meaning:
                    continue
                entry_key = f"{content}\n{meaning}"
                if entry_key in seen_entries:
                    continue
                seen_entries.add(entry_key)
                lines.append(f"- {content}: {meaning}")

        if not lines:
            return ""

        return "【黑话解释】\n" + "\n".join(lines[:8])

    def _build_context_jargon_explanation(
        self,
        chat_history: List[SessionMessage],
        reply_message: Optional[SessionMessage],
    ) -> str:
        """基于当前上下文自动匹配黑话。"""
        corpus = self._build_context_corpus(chat_history, reply_message)
        if not corpus:
            return ""

        jargon_records = self._load_jargon_records()
        lines: List[str] = []
        seen_contents: set[str] = set()
        for jargon in jargon_records:
            if not jargon.content or not jargon.meaning:
                continue
            if jargon.content in seen_contents:
                continue
            if not self._is_visible_jargon(jargon):
                continue
            if not self._is_jargon_in_corpus(jargon.content, corpus):
                continue

            seen_contents.add(jargon.content)
            lines.append(f"- {jargon.content}: {jargon.meaning}")
            if len(lines) >= 8:
                break

        if not lines:
            return ""

        logger.info(
            f"Built Maisaka jargon explanation: session_id={self._session_id} "
            f"count={len(lines)}"
        )
        return "【黑话解释】\n" + "\n".join(lines)

    def _load_jargon_records(self) -> List[_JargonRecord]:
        """在 session 内提取黑话的静态数据，避免 detached ORM 对象。"""
        with get_db_session(auto_commit=False) as session:
            query = select(Jargon).where(Jargon.meaning != "")  # type: ignore[attr-defined]
            query = query.order_by(Jargon.count.desc()).limit(200)  # type: ignore[attr-defined]
            jargons = session.exec(query).all()
            return [
                _JargonRecord(
                    jargon_id=jargon.id,
                    content=(jargon.content or "").strip(),
                    meaning=(jargon.meaning or "").strip(),
                    session_id_dict=jargon.session_id_dict or "{}",
                    is_global=bool(jargon.is_global),
                )
                for jargon in jargons
            ]

    def _build_context_corpus(
        self,
        chat_history: List[SessionMessage],
        reply_message: Optional[SessionMessage],
    ) -> str:
        """将最近上下文拼成待匹配文本。"""
        parts: List[str] = []
        for message in chat_history[-20:]:
            text = get_message_text(message).strip()
            if not text:
                continue
            _, body = parse_speaker_content(text)
            parts.append(body.strip() or text)

        if reply_message is not None:
            reply_text = get_message_text(reply_message).strip()
            if reply_text:
                _, body = parse_speaker_content(reply_text)
                parts.append(body.strip() or reply_text)

        return "\n".join(parts)

    def _is_visible_jargon(self, jargon: _JargonRecord) -> bool:
        """判断当前会话是否可见该黑话。"""
        if global_config.expression.all_global_jargon or jargon.is_global:
            return True

        try:
            session_id_dict = json.loads(jargon.session_id_dict or "{}")
        except (TypeError, json.JSONDecodeError):
            logger.warning(f"Failed to parse jargon.session_id_dict: jargon_id={jargon.jargon_id}")
            return False
        return self._session_id in session_id_dict

    @staticmethod
    def _is_jargon_in_corpus(content: str, corpus: str) -> bool:
        """判断黑话词条是否出现在上下文中。"""
        if re.search(r"[\u4e00-\u9fff]", content):
            return re.search(re.escape(content), corpus, flags=re.IGNORECASE) is not None

        pattern = rf"\b{re.escape(content)}\b"
        return re.search(pattern, corpus, flags=re.IGNORECASE) is not None
