from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
import time


class WorkflowStage(Enum):
    """Workflow阶段定义（MVP固定阶段）"""

    INGRESS = "ingress"
    PRE_PROCESS = "pre_process"
    PLAN = "plan"
    TOOL_EXECUTE = "tool_execute"
    POST_PROCESS = "post_process"
    EGRESS = "egress"

    def __str__(self) -> str:
        return self.value


@dataclass
class WorkflowContext:
    """Workflow上下文"""

    trace_id: str
    stream_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


@dataclass
class WorkflowMessage:
    """Workflow消息包装"""

    msg_type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, Any] = field(default_factory=dict)
    mutable_flags: Dict[str, bool] = field(default_factory=dict)


@dataclass
class WorkflowStepResult:
    """Workflow步骤结果"""

    status: Literal["continue", "stop", "failed"] = "continue"
    return_message: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass
class WorkflowStepInfo:
    """Workflow步骤元数据"""

    name: str
    stage: WorkflowStage
    plugin_name: str
    description: str = ""
    enabled: bool = True
    priority: int = 0
    timeout_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        return f"{self.plugin_name}.{self.name}"
