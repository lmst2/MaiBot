from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from PIL import Image as PILImage
from rich.traceback import install
from typing import Optional, List

import asyncio
import hashlib
import io
import traceback

from src.common.database.database_model import Images, ImageType
from src.common.logger import get_logger


install(extra_lines=3)

logger = get_logger("emoji")


class BaseImageDataModel(ABC):
    @classmethod
    @abstractmethod
    def from_db_instance(cls, image: "Images"):
        raise NotImplementedError

    @abstractmethod
    def to_db_instance(self) -> "Images":
        raise NotImplementedError

    def read_image_bytes(self, path: Path) -> bytes:
        """
        同步读取图片文件的字节内容
        
        Args:
            path (Path): 图片文件的完整路径
        Returns:
            return (bytes): 图片文件的字节内容
        Raises:
            FileNotFoundError: 如果文件不存在则抛出该异常
            Exception: 其他读取文件时发生的异常
        """
        try:
            with open(path, "rb") as f:
                return f.read()
        except FileNotFoundError as e:
            logger.error(f"[读取图片文件] 文件未找到: {path}")
            raise e
        except Exception as e:
            logger.error(f"[读取图片文件] 读取文件时发生错误: {e}")
            raise e

    def get_image_format(self, image_bytes: bytes) -> str:
        """
        获取图片的格式

        Args:
            image_bytes (bytes): 图片的字节内容

        Returns:
            return (str): 图片的格式（小写）

        Raises:
            ValueError: 如果无法识别图片格式
            Exception: 其他读取图片格式时发生的异常
        """
        try:
            with PILImage.open(io.BytesIO(image_bytes)) as img:
                if not img.format:
                    raise ValueError("无法识别图片格式")
                return img.format.lower()
        except Exception as e:
            logger.error(f"[获取图片格式] 读取图片格式时发生错误: {e}")
            raise e


class ImageDataModel(BaseImageDataModel):
    pass


class MaiEmoji(BaseImageDataModel):
    def __init__(self, full_path: str | Path):
        if not full_path:
            # 创建时候即检测文件路径合法性
            raise ValueError("表情包路径不能为空")
        if Path(full_path).is_dir() or not Path(full_path).exists():
            raise FileNotFoundError(f"表情包路径无效: {full_path}")
        resolved_path = Path(full_path).absolute().resolve()
        self.full_path: Path = resolved_path
        self.dir_path: Path = resolved_path.parent.resolve()
        self.file_name: str = resolved_path.name
        # self.embedding = []
        self.emoji_hash: str = None  # type: ignore
        self.description = ""
        self.emotion: List[str] = []
        self.query_count = 0
        self.register_time: Optional[datetime] = None
        self.last_used_time: Optional[datetime] = None

        # 私有属性
        self.is_deleted = False
        self._format: str = ""  # 图片格式

    @classmethod
    def from_db_instance(cls, image: Images):
        obj = cls(image.full_path)
        obj.emoji_hash = image.image_hash
        obj.description = image.description
        if image.emotion:
            obj.emotion = image.emotion.split(",")
        obj.query_count = image.query_count
        obj.last_used_time = image.last_used_time
        obj.register_time = image.register_time
        return obj

    def to_db_instance(self) -> Images:
        emotion_str = ",".join(self.emotion) if self.emotion else None
        return Images(
            image_hash=self.emoji_hash,
            description=self.description,
            full_path=str(self.full_path),
            image_type=ImageType.EMOJI,
            emotion=emotion_str,
            query_count=self.query_count,
            last_used_time=self.last_used_time,
            register_time=self.register_time,
        )

    async def calculate_hash_format(self) -> bool:
        """
        异步计算表情包的哈希值和格式
        
        Returns:
            return (bool): 如果成功计算哈希值和格式则返回True，否则返回False
        """
        logger.debug(f"[初始化] 正在读取文件: {self.full_path}")
        try:
            # 计算哈希值
            logger.debug(f"[初始化] 计算 {self.file_name} 的哈希值...")
            image_bytes = await asyncio.to_thread(self.read_image_bytes, self.full_path)
            self.emoji_hash = hashlib.sha256(image_bytes).hexdigest()
            logger.debug(f"[初始化] {self.file_name} 计算哈希值成功: {self.emoji_hash}")

            # 用PIL读取图片格式
            logger.debug(f"[初始化] 读取 {self.file_name} 的图片格式...")
            self._format = await asyncio.to_thread(self.get_image_format, image_bytes)
            logger.debug(f"[初始化] {self.file_name} 读取图片格式成功: {self._format}")

            # 比对文件扩展名和实际格式
            file_ext = self.file_name.split(".")[-1].lower()
            if file_ext != self._format:
                logger.warning(f"[初始化] {self.file_name} 文件扩展名与实际格式不符: ext`{file_ext}`!=`{self._format}`")
                # 重命名文件以匹配实际格式
                new_file_name = ".".join(self.file_name.split(".")[:-1] + [self._format])
                new_full_path = self.dir_path / new_file_name
                self.full_path.rename(new_full_path)
                self.full_path = new_full_path

            return True
        except Exception as e:
            logger.error(f"[初始化] 初始化表情包时发生错误: {e}")
            logger.error(traceback.format_exc())
            self.is_deleted = True
            return False
