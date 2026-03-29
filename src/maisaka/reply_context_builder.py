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

from .message_adapter import get_message_role, get_message_source, get_message_text, parse_speaker_content

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
    count: int
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
    ) -> str:
        """查询并格式化黑话解释。"""
        if not global_config.expression.enable_jargon_explanation:
            return ""

        return self._build_context_jargon_explanation(chat_history, reply_message)

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
        matched_records: List[tuple[int, int, int, _JargonRecord]] = []
        seen_contents: set[str] = set()
        for jargon in jargon_records:
            if not jargon.content or not jargon.meaning:
                continue
            normalized_content = jargon.content.lower()
            if normalized_content in seen_contents:
                continue
            if not self._is_visible_jargon(jargon):
                continue
            match_position = self._get_jargon_match_position(jargon.content, corpus)
            if match_position is None:
                continue

            seen_contents.add(normalized_content)
            matched_records.append((match_position, -len(jargon.content), -jargon.count, jargon))

        matched_records.sort()
        lines = [f"- {jargon.content}: {jargon.meaning}" for _, _, _, jargon in matched_records[:8]]

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
            query = select(Jargon).where(Jargon.is_jargon.is_(True), Jargon.meaning != "")  # type: ignore[attr-defined]
            query = query.order_by(Jargon.count.desc())  # type: ignore[attr-defined]
            jargons = session.exec(query).all()
            return [
                _JargonRecord(
                    jargon_id=jargon.id,
                    content=(jargon.content or "").strip(),
                    count=int(jargon.count or 0),
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
        """将当前聊天记录内所有用户消息拼成待匹配文本。"""
        parts: List[str] = []
        for message in chat_history:
            if get_message_role(message) != "user":
                continue
            if get_message_source(message) != "user":
                continue
            text = get_message_text(message).strip()
            if not text:
                continue
            _, body = parse_speaker_content(text)
            parts.append(body.strip() or text)

        if reply_message is not None and get_message_source(reply_message) == "user":
            reply_text = get_message_text(reply_message).strip()
            if reply_text:
                _, body = parse_speaker_content(reply_text)
                normalized_reply_text = body.strip() or reply_text
                if normalized_reply_text not in parts:
                    parts.append(normalized_reply_text)

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
    def _get_jargon_match_position(content: str, corpus: str) -> Optional[int]:
        """返回 jargon 在上下文中的首次命中位置，未命中时返回 `None`。"""
        if re.search(r"[\u4e00-\u9fff]", content):
            match = re.search(re.escape(content), corpus, flags=re.IGNORECASE)
            if match is None:
                return None
            return match.start()

        pattern = rf"\b{re.escape(content)}\b"
        match = re.search(pattern, corpus, flags=re.IGNORECASE)
        if match is None:
            return None
        return match.start()
