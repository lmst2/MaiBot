from typing import Optional

from src.common.database.database_model import Jargon

from . import BaseDatabaseDataModel


class MaiJargon(BaseDatabaseDataModel[Jargon]):
    """Jargon 数据模型，与数据库模型 Jargon 互转。"""

    def __init__(
        self,
        content: str,
        meaning: str,
        raw_content: Optional[str] = None,
        session_id: Optional[str] = None,
        count: int = 0,
        is_jargon: Optional[bool] = True,
        is_global: bool = False,
        is_complete: bool = False,
        inference_with_context: Optional[str] = None,
        inference_with_content_only: Optional[str] = None,
    ):
        self.content = content
        self.raw_content = raw_content
        self.meaning = meaning
        self.session_id = session_id
        self.count = count
        self.is_jargon = is_jargon
        self.is_global = is_global
        self.is_complete = is_complete
        self.inference_with_context = inference_with_context
        self.inference_with_content_only = inference_with_content_only

    @classmethod
    def from_db_instance(cls, db_record: Jargon) -> "MaiJargon":
        """从数据库模型创建 MaiJargon 实例。"""
        return cls(
            content=db_record.content,
            meaning=db_record.meaning,
            raw_content=db_record.raw_content,
            session_id=db_record.session_id,
            count=db_record.count,
            is_jargon=db_record.is_jargon,
            is_global=db_record.is_global,
            is_complete=db_record.is_complete,
            inference_with_context=db_record.inference_with_context,
            inference_with_content_only=db_record.inference_with_content_only,
        )

    def to_db_instance(self) -> Jargon:
        """将 MaiJargon 转换为数据库模型 Jargon。"""
        return Jargon(
            content=self.content,
            raw_content=self.raw_content,
            meaning=self.meaning,
            session_id=self.session_id,
            count=self.count,
            is_jargon=self.is_jargon,
            is_global=self.is_global,
            is_complete=self.is_complete,
            inference_with_context=self.inference_with_context,
            inference_with_content_only=self.inference_with_content_only,
        )
