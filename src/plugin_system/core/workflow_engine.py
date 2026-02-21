from typing import Any, Awaitable, Callable, List, Optional, Tuple, Union
import asyncio
import inspect
import time
import uuid

from src.common.logger import get_logger
from src.plugin_system.base.component_types import EventType, MaiMessages
from src.plugin_system.base.workflow_errors import WorkflowErrorCode
from src.plugin_system.base.workflow_types import WorkflowContext, WorkflowStage, WorkflowStepResult

logger = get_logger("workflow_engine")


class WorkflowEngine:
    """线性Workflow执行器（MVP）"""

    STAGE_EVENT_SEQUENCE: List[Tuple[WorkflowStage, Union[EventType, str]]] = [
        (WorkflowStage.INGRESS, "workflow.ingress"),
        (WorkflowStage.PRE_PROCESS, EventType.ON_MESSAGE_PRE_PROCESS),
        (WorkflowStage.PLAN, EventType.ON_PLAN),
        (WorkflowStage.TOOL_EXECUTE, "workflow.tool_execute"),
        (WorkflowStage.POST_PROCESS, EventType.POST_SEND_PRE_PROCESS),
        (WorkflowStage.EGRESS, "workflow.egress"),
    ]

    def __init__(self):
        self._execution_history: dict[str, dict[str, Any]] = {}

    async def execute_linear(
        self,
        dispatch_event: Callable[
            [Union[EventType, str], Optional[MaiMessages], Optional[str], Optional[List[str]]],
            Awaitable[Tuple[bool, Optional[MaiMessages]]],
        ],
        message: Optional[MaiMessages] = None,
        stream_id: Optional[str] = None,
        action_usage: Optional[List[str]] = None,
        context: Optional[WorkflowContext] = None,
    ) -> Tuple[WorkflowStepResult, Optional[MaiMessages], WorkflowContext]:
        """执行线性workflow。"""
        workflow_context = context or WorkflowContext(trace_id=uuid.uuid4().hex, stream_id=stream_id)
        current_message = message.deepcopy() if message else None
        self._execution_history[workflow_context.trace_id] = {
            "trace_id": workflow_context.trace_id,
            "stream_id": workflow_context.stream_id,
            "stages": [],
            "errors": [],
            "status": "running",
        }

        for stage, event_type in self.STAGE_EVENT_SEQUENCE:
            stage_key = str(stage)
            stage_start = time.perf_counter()
            try:
                should_continue, modified_message = await dispatch_event(
                    event_type,
                    current_message,
                    workflow_context.stream_id,
                    action_usage,
                )
                workflow_context.timings[stage_key] = time.perf_counter() - stage_start
                self._execution_history[workflow_context.trace_id]["stages"].append(
                    {
                        "stage": stage_key,
                        "event_type": str(event_type),
                        "event_continue": should_continue,
                        "event_cost": workflow_context.timings[stage_key],
                    }
                )

                if modified_message:
                    current_message = modified_message

                if not should_continue:
                    logger.info(f"[trace_id={workflow_context.trace_id}] Workflow在阶段 {stage_key} 被中断")
                    return (
                        WorkflowStepResult(
                            status="stop",
                            return_message=f"workflow stopped at stage {stage_key}",
                            diagnostics={
                                "stage": stage_key,
                                "trace_id": workflow_context.trace_id,
                                "error_code": WorkflowErrorCode.POLICY_BLOCKED.value,
                            },
                        ),
                        current_message,
                        workflow_context,
                    )

                step_result = await self._execute_registered_steps(stage, workflow_context, current_message)
                if step_result.status in ["stop", "failed"]:
                    self._execution_history[workflow_context.trace_id]["status"] = step_result.status
                    self._execution_history[workflow_context.trace_id]["errors"] = workflow_context.errors.copy()
                    return step_result, current_message, workflow_context
            except Exception as e:
                workflow_context.timings[stage_key] = time.perf_counter() - stage_start
                workflow_context.errors.append(f"{stage_key}: {e}")
                logger.error(f"[trace_id={workflow_context.trace_id}] Workflow阶段 {stage_key} 执行异常: {e}", exc_info=True)
                self._execution_history[workflow_context.trace_id]["status"] = "failed"
                self._execution_history[workflow_context.trace_id]["errors"] = workflow_context.errors.copy()
                return (
                    WorkflowStepResult(
                        status="failed",
                        return_message=str(e),
                        diagnostics={
                            "stage": stage_key,
                            "trace_id": workflow_context.trace_id,
                            "error_code": WorkflowErrorCode.DOWNSTREAM_FAILED.value,
                        },
                    ),
                    current_message,
                    workflow_context,
                )

        self._execution_history[workflow_context.trace_id]["status"] = "continue"
        self._execution_history[workflow_context.trace_id]["errors"] = workflow_context.errors.copy()
        return (
            WorkflowStepResult(
                status="continue",
                return_message="workflow completed",
                diagnostics={"trace_id": workflow_context.trace_id},
            ),
            current_message,
            workflow_context,
        )

    async def _execute_registered_steps(
        self,
        stage: WorkflowStage,
        context: WorkflowContext,
        message: Optional[MaiMessages],
    ) -> WorkflowStepResult:
        """执行指定阶段已注册的workflow步骤。"""
        from src.plugin_system.core.component_registry import component_registry

        stage_steps = component_registry.get_steps_by_stage(stage, enabled_only=True)
        sorted_steps = sorted(stage_steps.values(), key=lambda step_info: step_info.priority, reverse=True)

        for step_info in sorted_steps:
            handler = component_registry.get_workflow_step_handler(step_info.full_name, stage)
            if not handler:
                context.errors.append(f"{step_info.full_name}: handler not found")
                continue

            step_timing_key = f"{stage.value}:{step_info.full_name}"
            step_start = time.perf_counter()
            timeout_seconds = step_info.timeout_ms / 1000 if step_info.timeout_ms > 0 else None

            try:
                if inspect.iscoroutinefunction(handler):
                    coroutine = handler(context, message)
                    result = await asyncio.wait_for(coroutine, timeout_seconds) if timeout_seconds else await coroutine
                else:
                    if timeout_seconds:
                        result = await asyncio.wait_for(asyncio.to_thread(handler, context, message), timeout_seconds)
                    else:
                        result = handler(context, message)
                    if inspect.isawaitable(result):
                        result = await asyncio.wait_for(result, timeout_seconds) if timeout_seconds else await result
                context.timings[step_timing_key] = time.perf_counter() - step_start

                normalized_result = self._normalize_step_result(result)
                if normalized_result.status == "continue":
                    continue

                normalized_result.diagnostics.setdefault("stage", stage.value)
                normalized_result.diagnostics.setdefault("step", step_info.full_name)
                normalized_result.diagnostics.setdefault("trace_id", context.trace_id)
                if normalized_result.status == "failed":
                    context.errors.append(
                        f"{step_info.full_name}: {normalized_result.return_message or 'workflow step failed'}"
                    )
                    normalized_result.diagnostics.setdefault("error_code", WorkflowErrorCode.DOWNSTREAM_FAILED.value)
                return normalized_result

            except asyncio.TimeoutError:
                context.timings[step_timing_key] = time.perf_counter() - step_start
                timeout_message = f"workflow step timeout after {step_info.timeout_ms}ms"
                context.errors.append(f"{step_info.full_name}: {timeout_message}")
                logger.error(
                    f"[trace_id={context.trace_id}] Workflow step {step_info.full_name} 超时: {timeout_message}"
                )
                return WorkflowStepResult(
                    status="failed",
                    return_message=timeout_message,
                    diagnostics={
                        "stage": stage.value,
                        "step": step_info.full_name,
                        "trace_id": context.trace_id,
                        "error_code": WorkflowErrorCode.STEP_TIMEOUT.value,
                    },
                )

            except Exception as e:
                context.timings[step_timing_key] = time.perf_counter() - step_start
                context.errors.append(f"{step_info.full_name}: {e}")
                logger.error(f"[trace_id={context.trace_id}] Workflow step {step_info.full_name} 执行异常: {e}", exc_info=True)
                return WorkflowStepResult(
                    status="failed",
                    return_message=str(e),
                    diagnostics={
                        "stage": stage.value,
                        "step": step_info.full_name,
                        "trace_id": context.trace_id,
                        "error_code": WorkflowErrorCode.DOWNSTREAM_FAILED.value,
                    },
                )

        return WorkflowStepResult(status="continue", diagnostics={"stage": stage.value, "trace_id": context.trace_id})

    def _normalize_step_result(self, result: Any) -> WorkflowStepResult:
        """归一化workflow step返回值。"""
        if isinstance(result, WorkflowStepResult):
            return result
        if isinstance(result, bool):
            if result:
                return WorkflowStepResult(status="continue")
            return WorkflowStepResult(
                status="failed",
                diagnostics={"error_code": WorkflowErrorCode.DOWNSTREAM_FAILED.value},
            )
        if result is None:
            return WorkflowStepResult(status="continue")
        if isinstance(result, str):
            return WorkflowStepResult(status="continue", return_message=result)
        if isinstance(result, dict):
            status = result.get("status", "continue")
            if status not in ["continue", "stop", "failed"]:
                status = "failed"
            return WorkflowStepResult(
                status=status,
                return_message=result.get("return_message"),
                diagnostics=result.get("diagnostics", {}),
                events=result.get("events", []),
            )
        return WorkflowStepResult(
            status="failed",
            return_message=f"unsupported step result type: {type(result)}",
            diagnostics={"error_code": WorkflowErrorCode.BAD_PAYLOAD.value},
        )

    def get_execution_trace(self, trace_id: str) -> Optional[dict[str, Any]]:
        """按trace_id获取workflow执行路径。"""
        trace = self._execution_history.get(trace_id)
        return trace.copy() if trace else None

    def clear_execution_trace(self, trace_id: str) -> bool:
        """清理trace执行记录。"""
        if trace_id in self._execution_history:
            del self._execution_history[trace_id]
            return True
        return False


workflow_engine = WorkflowEngine()
