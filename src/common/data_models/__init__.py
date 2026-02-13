from abc import ABC, abstractmethod

from dataclasses import is_dataclass
from typing import Any, Dict, Self, TypeVar, Generic, TYPE_CHECKING

import copy

if TYPE_CHECKING:
    from sqlmodel import SQLModel

T = TypeVar("T", bound="SQLModel")


class BaseDataModel:
    def deepcopy(self):
        return copy.deepcopy(self)


def transform_class_to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if is_dataclass(obj):
        return obj.__dict__
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return {"value": obj}


class BaseDatabaseDataModel(ABC, Generic[T]):
    @classmethod
    @abstractmethod
    def from_db_instance(cls, db_record: T) -> Self:
        """从数据库实例创建数据模型对象"""
        raise NotImplementedError

    @abstractmethod
    def to_db_instance(self) -> T:
        """将数据模型对象转换为数据库实例"""
        raise NotImplementedError
