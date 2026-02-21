from enum import Enum


class WorkflowErrorCode(Enum):
    """Workflow统一错误码"""

    PLUGIN_NOT_READY = "PLUGIN_NOT_READY"
    STEP_TIMEOUT = "STEP_TIMEOUT"
    BAD_PAYLOAD = "BAD_PAYLOAD"
    DOWNSTREAM_FAILED = "DOWNSTREAM_FAILED"
    POLICY_BLOCKED = "POLICY_BLOCKED"

    def __str__(self) -> str:
        return self.value
