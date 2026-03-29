"""
MaiSaka knowledge store.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import json

# 数据目录位于项目根目录下的 mai_knowledge
PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DATA_DIR = PROJECT_ROOT / "mai_knowledge"
KNOWLEDGE_FILE = KNOWLEDGE_DATA_DIR / "knowledge.json"


KNOWLEDGE_CATEGORIES = {
    "1": "性别",
    "2": "性格",
    "3": "饮食口味",
    "4": "交友喜好",
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
    """
    简单的 Maisaka 知识存储。

    特性：
    - 持久化到 JSON 文件
    - 按分类存储用户画像类知识
    - 支持基础去重
    """

    def __init__(self) -> None:
        """初始化知识存储。"""
        self._knowledge: Dict[str, List[Dict[str, Any]]] = {
            category_id: [] for category_id in KNOWLEDGE_CATEGORIES
        }
        self._ensure_data_dir()
        self._load()

    def _ensure_data_dir(self) -> None:
        """确保数据目录存在。"""
        KNOWLEDGE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        """从文件加载知识数据。"""
        if not KNOWLEDGE_FILE.exists():
            self._knowledge = {category_id: [] for category_id in KNOWLEDGE_CATEGORIES}
            return

        try:
            with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as file:
                loaded = json.load(file)

            normalized_knowledge: Dict[str, List[Dict[str, Any]]] = {
                category_id: [] for category_id in KNOWLEDGE_CATEGORIES
            }
            for category_id in KNOWLEDGE_CATEGORIES:
                category_items = loaded.get(category_id, [])
                if isinstance(category_items, list):
                    normalized_knowledge[category_id] = [
                        item for item in category_items if isinstance(item, dict)
                    ]
            self._knowledge = normalized_knowledge
        except Exception:
            self._knowledge = {category_id: [] for category_id in KNOWLEDGE_CATEGORIES}

    def _save(self) -> None:
        """保存知识数据到文件。"""
        with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as file:
            json.dump(self._knowledge, file, ensure_ascii=False, indent=2)

    @staticmethod
    def _normalize_content(content: str) -> str:
        """标准化知识内容，便于去重。"""
        return " ".join(str(content).strip().split())

    def add_knowledge(
        self,
        category_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        添加一条知识信息。

        Args:
            category_id: 分类编号
            content: 知识内容
            metadata: 附加元数据

        Returns:
            是否新增成功；若命中去重则返回 False
        """
        if category_id not in KNOWLEDGE_CATEGORIES:
            return False

        normalized_content = self._normalize_content(content)
        if not normalized_content:
            return False

        existing_items = self._knowledge.get(category_id, [])
        for item in existing_items:
            existing_content = self._normalize_content(str(item.get("content", "")))
            if existing_content == normalized_content:
                return False

        knowledge_item = {
            "id": f"know_{category_id}_{datetime.now().timestamp()}",
            "content": normalized_content,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
        }
        self._knowledge[category_id].append(knowledge_item)
        self._save()
        return True

    def get_category_knowledge(self, category_id: str) -> List[Dict[str, Any]]:
        """获取某个分类下的所有知识。"""
        return self._knowledge.get(category_id, [])

    def get_all_knowledge(self) -> Dict[str, List[Dict[str, Any]]]:
        """获取全部知识。"""
        return self._knowledge

    def get_category_name(self, category_id: str) -> str:
        """获取分类名称。"""
        return KNOWLEDGE_CATEGORIES.get(category_id, "未知分类")

    def get_categories_summary(self) -> str:
        """获取分类摘要，供模型判断是否需要检索。"""
        lines: List[str] = []
        for category_id, category_name in KNOWLEDGE_CATEGORIES.items():
            count = len(self._knowledge.get(category_id, []))
            count_text = f"{count}条" if count > 0 else "无数据"
            lines.append(f"{category_id}. {category_name} ({count_text})")
        return "\n".join(lines)

    def get_formatted_knowledge(self, category_ids: List[str], limit_per_category: int = 5) -> str:
        """
        获取指定分类的格式化知识内容。

        Args:
            category_ids: 分类编号列表
            limit_per_category: 每个分类最多返回多少条

        Returns:
            格式化后的知识内容
        """
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
        total_items = sum(len(items) for items in self._knowledge.values())
        return {
            "total_categories": len(KNOWLEDGE_CATEGORIES),
            "total_items": total_items,
            "data_file": str(KNOWLEDGE_FILE),
            "data_exists": KNOWLEDGE_FILE.exists(),
            "data_size_kb": KNOWLEDGE_FILE.stat().st_size / 1024 if KNOWLEDGE_FILE.exists() else 0,
        }


_knowledge_store_instance: Optional[KnowledgeStore] = None


def get_knowledge_store() -> KnowledgeStore:
    """获取知识存储单例。"""
    global _knowledge_store_instance
    if _knowledge_store_instance is None:
        _knowledge_store_instance = KnowledgeStore()
    return _knowledge_store_instance
