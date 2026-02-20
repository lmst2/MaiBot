"""
插件核心管理模块

提供插件的加载、注册和管理功能
"""

from src.plugin_system.core.plugin_manager import plugin_manager
from src.plugin_system.core.component_registry import component_registry
from src.plugin_system.core.events_manager import events_manager
from src.plugin_system.core.global_announcement_manager import global_announcement_manager
from src.plugin_system.core.plugin_service_registry import plugin_service_registry
from src.plugin_system.core.workflow_engine import workflow_engine

__all__ = [
    "plugin_manager",
    "component_registry",
    "events_manager",
    "global_announcement_manager",
    "plugin_service_registry",
    "workflow_engine",
]
