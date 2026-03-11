from datetime import datetime
from typing import Optional, Dict

import json

from src.common.database.database_model import ActionRecord

from . import BaseDatabaseDataModel


class MaiActionRecord(BaseDatabaseDataModel[ActionRecord]):
    def __init__(
        self,
        action_id: str,
        timestamp: datetime,
        session_id: str,
        action_name: str,
        action_reasoning: Optional[str] = None,
        action_data: Optional[Dict] = None,
        action_builtin_prompt: Optional[str] = None,
        action_display_prompt: Optional[str] = None,
    ):
        self.action_id = action_id
        """动作ID"""
        self.timestamp = timestamp
        """时间戳"""
        self.session_id = session_id
        """会话ID"""
        self.action_name = action_name
        """动作名称"""
        self.action_reasoning = action_reasoning
        """动作推理过程"""
        self.action_data = action_data or {}
        """动作数据"""
        self.action_builtin_prompt = action_builtin_prompt
        """内置动作提示"""
        self.action_display_prompt = action_display_prompt
        """最终输入到 Prompt 的内容"""

    @classmethod
    def from_db_instance(cls, db_record: ActionRecord):
        """Create a data model object from a database record."""
        return cls(
            action_id=db_record.action_id,
            timestamp=db_record.timestamp,
            session_id=db_record.session_id,
            action_name=db_record.action_name,
            action_reasoning=db_record.action_reasoning,
            action_data=json.loads(db_record.action_data) if db_record.action_data else None,
            action_builtin_prompt=db_record.action_builtin_prompt,
            action_display_prompt=db_record.action_display_prompt,
        )

    def to_db_instance(self):
        """Convert the data model object back to a database instance."""
        return ActionRecord(
            action_id=self.action_id,
            timestamp=self.timestamp,
            session_id=self.session_id,
            action_name=self.action_name,
            action_reasoning=self.action_reasoning,
            action_data=json.dumps(self.action_data) if self.action_data else None,
            action_builtin_prompt=self.action_builtin_prompt,
            action_display_prompt=self.action_display_prompt,
        )
