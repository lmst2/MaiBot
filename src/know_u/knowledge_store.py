"""
MaiSaka knowledge store.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import json

from sqlmodel import col, select

from src.common.database.database import DATABASE_URL, get_db_session
from src.common.database.database_model import MaiKnowledge

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DATA_DIR = PROJECT_ROOT / "mai_knowledge"
KNOWLEDGE_FILE = KNOWLEDGE_DATA_DIR / "knowledge.json"


KNOWLEDGE_CATEGORIES = {
    "1": "性别",
    "2": "性格",
    "3": "饮食口味",
    "4": "交友偏好",
    "5": "情绪/理性倾向",
    "6": "兴趣爱好",
    "7": "职业/专业",
    "8": "生活习惯",
    "9": "价值观",
    "10": "沟通风格",
    "11": "学习方式",
    "12": "压力应对方式",
}


class KnowledgeStore:
    """存储 Maisaka 的用户画像知识。"""

    def __init__(self) -> None:
        """初始化知识存储，并在需要时迁移旧版 JSON 数据。"""
        self._ensure_legacy_data_dir()
        self._migrate_legacy_file_if_needed()

    def _ensure_legacy_data_dir(self) -> None:
        """确保旧版知识目录存在，便于兼容历史数据。"""
        KNOWLEDGE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_content(content: str) -> str:
        """标准化知识内容，便于去重。"""
        return " ".join(str(content).strip().split())

    @staticmethod
    def _serialize_metadata(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
        """将元数据序列化为 JSON 文本。"""
        if not metadata:
            return None
        return json.dumps(metadata, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _deserialize_metadata(raw_text: Optional[str]) -> Dict[str, Any]:
        """将 JSON 文本反序列化为元数据字典。"""
        if not raw_text:
            return {}
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _parse_created_at(raw_value: Any) -> datetime:
        """解析旧版数据中的创建时间。"""
        if isinstance(raw_value, datetime):
            return raw_value
        if isinstance(raw_value, str):
            raw_text = raw_value.strip()
            if raw_text:
                try:
                    return datetime.fromisoformat(raw_text)
                except ValueError:
                    pass
        return datetime.now()

    @classmethod
    def _build_item_dict(cls, record: MaiKnowledge) -> Dict[str, Any]:
        """将数据库记录转换为兼容旧接口的字典。"""
        return {
            "id": record.knowledge_id,
            "content": record.content,
            "metadata": cls._deserialize_metadata(record.metadata_json),
            "created_at": record.created_at.isoformat(),
        }

    def _load_legacy_knowledge_file(self) -> Dict[str, List[Dict[str, Any]]]:
        """读取旧版 JSON 知识文件。"""
        if not KNOWLEDGE_FILE.exists():
            return {}

        try:
            with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as file:
                loaded = json.load(file)
        except Exception:
            return {}

        if not isinstance(loaded, dict):
            return {}

        normalized_knowledge: Dict[str, List[Dict[str, Any]]] = {}
        for category_id in KNOWLEDGE_CATEGORIES:
            category_items = loaded.get(category_id, [])
            if isinstance(category_items, list):
                normalized_knowledge[category_id] = [
                    item for item in category_items if isinstance(item, dict)
                ]
        return normalized_knowledge

    def _migrate_legacy_file_if_needed(self) -> None:
        """在数据库为空时，将旧版 JSON 中的知识导入数据库。"""
        legacy_knowledge = self._load_legacy_knowledge_file()
        if not legacy_knowledge:
            return

        with get_db_session(auto_commit=False) as session:
            existing_record = session.exec(select(MaiKnowledge.id).limit(1)).first()
            if existing_record is not None:
                return

            for category_id, items in legacy_knowledge.items():
                if category_id not in KNOWLEDGE_CATEGORIES:
                    continue

                for item in items:
                    content = self._normalize_content(str(item.get("content", "")))
                    if not content:
                        continue

                    metadata = item.get("metadata")
                    session.add(
                        MaiKnowledge(
                            knowledge_id=str(item.get("id") or f"know_{category_id}_{datetime.now().timestamp()}"),
                            category_id=category_id,
                            content=content,
                            normalized_content=content,
                            metadata_json=self._serialize_metadata(metadata if isinstance(metadata, dict) else None),
                            created_at=self._parse_created_at(item.get("created_at")),
                        )
                    )

            session.commit()

    def add_knowledge(
        self,
        category_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """添加一条知识信息。"""
        if category_id not in KNOWLEDGE_CATEGORIES:
            return False

        normalized_content = self._normalize_content(content)
        if not normalized_content:
            return False

        user_platform = str((metadata or {}).get("platform", "")).strip()
        user_id = str((metadata or {}).get("user_id", "")).strip()
        with get_db_session(auto_commit=False) as session:
            existing_records = session.exec(
                select(MaiKnowledge).where(
                    MaiKnowledge.category_id == category_id,
                    MaiKnowledge.normalized_content == normalized_content,
                )
            ).all()
            for existing_record in existing_records:
                existing_metadata = self._deserialize_metadata(existing_record.metadata_json)
                existing_platform = str(existing_metadata.get("platform", "")).strip()
                existing_user_id = str(existing_metadata.get("user_id", "")).strip()
                if user_platform and user_id:
                    if existing_platform == user_platform and existing_user_id == user_id:
                        return False
                    continue
                if not existing_platform and not existing_user_id:
                    return False

            session.add(
                MaiKnowledge(
                    knowledge_id=f"know_{category_id}_{datetime.now().timestamp()}",
                    category_id=category_id,
                    content=normalized_content,
                    normalized_content=normalized_content,
                    metadata_json=self._serialize_metadata(metadata),
                    created_at=datetime.now(),
                )
            )
            session.commit()
        return True

    def search_knowledge(
        self,
        keyword: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """按关键词搜索知识内容。"""
        normalized_keyword = self._normalize_content(keyword)
        if not normalized_keyword:
            return []

        limit_value = max(1, int(limit))
        with get_db_session() as session:
            records = session.exec(
                select(MaiKnowledge)
                .where(
                    col(MaiKnowledge.content).contains(normalized_keyword)
                    | col(MaiKnowledge.normalized_content).contains(normalized_keyword)
                )
                .order_by(MaiKnowledge.created_at.desc(), MaiKnowledge.id.desc())
                .limit(limit_value)
            ).all()

        results: List[Dict[str, Any]] = []
        for record in records:
            item = self._build_item_dict(record)
            item["category_id"] = record.category_id
            item["category_name"] = self.get_category_name(record.category_id)
            results.append(item)
        return results

    def get_knowledge_by_user(
        self,
        *,
        platform: str = "",
        user_id: str = "",
        user_nickname: str = "",
        person_name: str = "",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """按用户元信息筛选知识条目。"""
        platform = str(platform).strip()
        user_id = str(user_id).strip()
        user_nickname = str(user_nickname).strip()
        person_name = str(person_name).strip()
        if not any((platform, user_id, user_nickname, person_name)):
            return []

        limit_value = max(1, int(limit))
        with get_db_session() as session:
            records = session.exec(
                select(MaiKnowledge).order_by(MaiKnowledge.created_at.desc(), MaiKnowledge.id.desc())
            ).all()

        results: List[Dict[str, Any]] = []
        for record in records:
            metadata = self._deserialize_metadata(record.metadata_json)
            if user_id and str(metadata.get("user_id", "")).strip() != user_id:
                continue
            if platform and str(metadata.get("platform", "")).strip() != platform:
                continue
            if user_nickname and str(metadata.get("user_nickname", "")).strip() != user_nickname:
                continue
            if person_name and str(metadata.get("person_name", "")).strip() != person_name:
                continue

            item = self._build_item_dict(record)
            item["category_id"] = record.category_id
            item["category_name"] = self.get_category_name(record.category_id)
            results.append(item)
            if len(results) >= limit_value:
                break

        return results

    def get_category_knowledge(self, category_id: str) -> List[Dict[str, Any]]:
        """获取某个分类下的所有知识。"""
        if category_id not in KNOWLEDGE_CATEGORIES:
            return []

        with get_db_session() as session:
            records = session.exec(
                select(MaiKnowledge)
                .where(MaiKnowledge.category_id == category_id)
                .order_by(MaiKnowledge.created_at.asc(), MaiKnowledge.id.asc())
            ).all()
        return [self._build_item_dict(record) for record in records]

    def get_all_knowledge(self) -> Dict[str, List[Dict[str, Any]]]:
        """获取全部知识。"""
        all_knowledge: Dict[str, List[Dict[str, Any]]] = {
            category_id: [] for category_id in KNOWLEDGE_CATEGORIES
        }
        with get_db_session() as session:
            records = session.exec(
                select(MaiKnowledge).order_by(
                    MaiKnowledge.category_id.asc(),
                    MaiKnowledge.created_at.asc(),
                    MaiKnowledge.id.asc(),
                )
            ).all()

        for record in records:
            all_knowledge.setdefault(record.category_id, []).append(self._build_item_dict(record))
        return all_knowledge

    def get_category_name(self, category_id: str) -> str:
        """获取分类名称。"""
        return KNOWLEDGE_CATEGORIES.get(category_id, "未知分类")

    def get_categories_summary(self) -> str:
        """获取分类摘要，供模型判断是否需要检索。"""
        counts: Dict[str, int] = {category_id: 0 for category_id in KNOWLEDGE_CATEGORIES}
        with get_db_session() as session:
            records = session.exec(select(MaiKnowledge.category_id)).all()

        for category_id in records:
            if category_id in counts:
                counts[category_id] += 1

        lines: List[str] = []
        for category_id, category_name in KNOWLEDGE_CATEGORIES.items():
            count = counts.get(category_id, 0)
            count_text = f"{count}条" if count > 0 else "无数据"
            lines.append(f"{category_id}. {category_name} ({count_text})")
        return "\n".join(lines)

    def get_formatted_knowledge(self, category_ids: List[str], limit_per_category: int = 5) -> str:
        """获取指定分类的格式化知识内容。"""
        parts: List[str] = []
        for category_id in category_ids:
            items = self.get_category_knowledge(category_id)
            if not items:
                continue

            category_name = self.get_category_name(category_id)
            parts.append(f"【{category_name}】")

            recent_items = items[-limit_per_category:]
            for item in recent_items:
                content = str(item.get("content", "")).strip()
                if content:
                    parts.append(f"- {content}")

        return "\n".join(parts)

    def get_stats(self) -> Dict[str, Any]:
        """获取知识数据统计。"""
        with get_db_session() as session:
            total_items = len(session.exec(select(MaiKnowledge.id)).all())

        return {
            "total_categories": len(KNOWLEDGE_CATEGORIES),
            "total_items": total_items,
            "data_file": DATABASE_URL,
            "data_exists": True,
            "data_size_kb": 0,
            "legacy_data_file": str(KNOWLEDGE_FILE),
            "legacy_data_exists": KNOWLEDGE_FILE.exists(),
            "storage_type": "database",
        }


_knowledge_store_instance: Optional[KnowledgeStore] = None


def get_knowledge_store() -> KnowledgeStore:
    """获取知识存储单例。"""
    global _knowledge_store_instance
    if _knowledge_store_instance is None:
        _knowledge_store_instance = KnowledgeStore()
    return _knowledge_store_instance
