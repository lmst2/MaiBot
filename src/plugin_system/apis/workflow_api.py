from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from src.plugin_system.base.component_types import EventType, MaiMessages
from src.plugin_system.base.workflow_types import WorkflowContext, WorkflowStage, WorkflowStepInfo, WorkflowStepResult
from src.plugin_system.core.component_registry import component_registry
from src.plugin_system.core.events_manager import events_manager
from src.plugin_system.core.workflow_engine import workflow_engine


def register_workflow_step(step_info: WorkflowStepInfo, step_handler: Callable[..., Any]) -> bool:
    """注册workflow step。"""
    return component_registry.register_workflow_step(step_info, step_handler)


def get_steps_by_stage(stage: WorkflowStage, enabled_only: bool = False) -> Dict[str, WorkflowStepInfo]:
    """获取指定阶段的workflow steps。"""
    return component_registry.get_steps_by_stage(stage, enabled_only=enabled_only)


def get_workflow_step(step_name: str, stage: Optional[WorkflowStage] = None) -> Optional[WorkflowStepInfo]:
    """获取workflow step元信息。"""
    return component_registry.get_workflow_step(step_name, stage)


def get_workflow_step_handler(step_name: str, stage: Optional[WorkflowStage] = None) -> Optional[Callable[..., Any]]:
    """获取workflow step处理函数。"""
    return component_registry.get_workflow_step_handler(step_name, stage)


def enable_workflow_step(step_name: str, stage: Optional[WorkflowStage] = None) -> bool:
    """启用workflow step。"""
    return component_registry.enable_workflow_step(step_name, stage)


def disable_workflow_step(step_name: str, stage: Optional[WorkflowStage] = None) -> bool:
    """禁用workflow step。"""
    return component_registry.disable_workflow_step(step_name, stage)


def get_execution_trace(trace_id: str) -> Optional[Dict[str, Any]]:
    """按trace_id获取workflow执行路径。"""
    return workflow_engine.get_execution_trace(trace_id)


def clear_execution_trace(trace_id: str) -> bool:
    """清理trace执行路径记录。"""
    return workflow_engine.clear_execution_trace(trace_id)


async def execute_workflow_message(
    message: Optional[MaiMessages] = None,
    stream_id: Optional[str] = None,
    action_usage: Optional[List[str]] = None,
    context: Optional[WorkflowContext] = None,
) -> Tuple[WorkflowStepResult, Optional[MaiMessages], WorkflowContext]:
    """执行workflow消息流转。"""
    return await events_manager.handle_workflow_message(
        message=message,
        stream_id=stream_id,
        action_usage=action_usage,
        context=context,
    )


async def publish_event(
    event_type: Union[EventType, str],
    message: Optional[MaiMessages] = None,
    stream_id: Optional[str] = None,
    action_usage: Optional[List[str]] = None,
) -> Tuple[bool, Optional[MaiMessages]]:
    """发布事件（支持系统事件和自定义字符串事件）。"""
    return await events_manager.handle_mai_events(
        event_type=event_type,
        message=message,
        stream_id=stream_id,
        action_usage=action_usage,
    )
