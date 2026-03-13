"""
MaiSaka - 了解列表持久化存储
存储用户个人特征信息，支持层级结构和本地持久化。
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# 数据目录 - 项目根目录下的 mai_knowledge
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_DATA_DIR = PROJECT_ROOT / "mai_knowledge"
KNOWLEDGE_FILE = KNOWLEDGE_DATA_DIR / "knowledge.json"


# 个人特征分类列表（预定义）
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
    了解列表存储。

    特性：
    - 持久化到 JSON 文件
    - 层级结构存储（按分类）
    - 支持增量更新
    - 启动时自动加载
    """

    def __init__(self):
        """初始化了解存储"""
        self._knowledge: Dict[str, List[Dict[str, Any]]] = {category_id: [] for category_id in KNOWLEDGE_CATEGORIES}
        self._ensure_data_dir()
        self._load()

    def _ensure_data_dir(self):
        """确保数据目录存在"""
        KNOWLEDGE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _load(self):
        """从文件加载了解数据"""
        if not KNOWLEDGE_FILE.exists():
            self._knowledge = {category_id: [] for category_id in KNOWLEDGE_CATEGORIES}
            return

        try:
            with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                # 确保所有分类都存在
                for category_id in KNOWLEDGE_CATEGORIES:
                    if category_id not in loaded:
                        loaded[category_id] = []
                self._knowledge = loaded
        except Exception as e:
            print(f"[warning]加载了解数据失败: {e}[/warning]")
            self._knowledge = {category_id: [] for category_id in KNOWLEDGE_CATEGORIES}

    def _save(self):
        """保存了解数据到文件"""
        try:
            with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
                json.dump(self._knowledge, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[warning]保存了解数据失败: {e}[/warning]")

    def add_knowledge(
        self,
        category_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        添加一条了解信息。

        Args:
            category_id: 分类编号
            content: 了解内容
            metadata: 元数据

        Returns:
            是否添加成功
        """
        if category_id not in KNOWLEDGE_CATEGORIES:
            return False

        try:
            knowledge_item = {
                "id": f"know_{category_id}_{datetime.now().timestamp()}",
                "content": content,
                "metadata": metadata or {},
                "created_at": datetime.now().isoformat(),
            }
            self._knowledge[category_id].append(knowledge_item)
            self._save()
            return True
        except Exception:
            return False

    def get_category_knowledge(self, category_id: str) -> List[Dict[str, Any]]:
        """
        获取某个分类的所有了解信息。

        Args:
            category_id: 分类编号

        Returns:
            该分类的所有了解信息
        """
        return self._knowledge.get(category_id, [])

    def get_all_knowledge(self) -> Dict[str, List[Dict[str, Any]]]:
        """获取所有了解信息"""
        return self._knowledge

    def get_category_name(self, category_id: str) -> str:
        """获取分类名称"""
        return KNOWLEDGE_CATEGORIES.get(category_id, "未知分类")

    def get_categories_summary(self) -> str:
        """获取所有分类的摘要（用于 LLM 展示）"""
        lines = []
        for category_id, category_name in KNOWLEDGE_CATEGORIES.items():
            count = len(self._knowledge.get(category_id, []))
            if count > 0:
                lines.append(f"{category_id}. {category_name} ({count}条)")
            else:
                lines.append(f"{category_id}. {category_name} (无数据)")
        return "\n".join(lines)

    def get_formatted_knowledge(self, category_ids: List[str]) -> str:
        """
        获取指定分类的了解内容，格式化为文本。

        Args:
            category_ids: 分类编号列表

        Returns:
            格式化后的了解内容文本
        """
        parts = []
        for category_id in category_ids:
            category_name = self.get_category_name(category_id)
            items = self.get_category_knowledge(category_id)

            if items:
                parts.append(f"【{category_name}】")
                for item in items:
                    content = item.get("content", "")
                    parts.append(f"  - {content}")

        return "\n".join(parts) if parts else "暂无相关了解信息"

    def get_stats(self) -> Dict[str, Any]:
        """获取了解数据统计信息"""
        total_items = sum(len(items) for items in self._knowledge.values())
        return {
            "total_categories": len(KNOWLEDGE_CATEGORIES),
            "total_items": total_items,
            "data_file": str(KNOWLEDGE_FILE),
            "data_exists": KNOWLEDGE_FILE.exists(),
            "data_size_kb": KNOWLEDGE_FILE.stat().st_size / 1024 if KNOWLEDGE_FILE.exists() else 0,
        }


# 全局单例
_knowledge_store_instance: Optional[KnowledgeStore] = None


def get_knowledge_store() -> KnowledgeStore:
    """获取了解存储实例（单例模式）"""
    global _knowledge_store_instance
    if _knowledge_store_instance is None:
        _knowledge_store_instance = KnowledgeStore()
    return _knowledge_store_instance
