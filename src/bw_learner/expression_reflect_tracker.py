from typing import TYPE_CHECKING, Optional

import time

from src.common.logger import get_logger
from src.common.database.database import get_db_session
from src.llm_models.utils_model import LLMRequest
from src.config.config import model_config

if TYPE_CHECKING:
    from src.common.data_models.expression_data_model import MaiExpression

# TODO: 这个LLMRequest实例被更优雅的方式替换掉
judge_model = LLMRequest(model_set=model_config.model_task_config.tool_use, request_type="reflect.tracker")

logger = get_logger("reflect_tracker")

class ReflectTracker:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.last_check_msg_count = 0
        self.max_msg_count = 30
        self.max_duration = 15 * 60  # 15 分钟
        self.expression: Optional["MaiExpression"] = None  # 当前正在追踪的表达，由外部设置

        # 运行状态
        self.tracking = False
        self.tracking_start_time: float = 0.0

    def register_expression_and_track(self, expression: "MaiExpression"):
        """注册需要追踪的表达"""
        if self.tracking:
            raise RuntimeError("ReflectTracker is already tracking an expression.")
        self.expression = expression
        self.tracking = True
        self.tracking_start_time = time.time()
    
    def _reset_tracker(self):
        """重置追踪状态"""
        self.expression = None
        self.tracking = False
        self.last_check_msg_count = 0

    async def trigger_tracker(self) -> bool:
        """
        触发追踪检查

        Returns:
            return (bool): 如果返回True，表示追踪完成，Tracker运行结束（运行状态置为`False`）；如果返回False，表示继续追踪
        """
        # 对于没有正在追踪的表达，直接返回False
        if not self.tracking:
            return False

        # 检查是否超时（无论是消息数量还是时间）
        if time.time() - self.tracking_start_time > self.max_duration:
            self._reset_tracker()
            return True
        
        # TODO: 完成追踪检查逻辑