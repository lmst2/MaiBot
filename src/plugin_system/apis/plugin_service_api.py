from typing import Any, Callable, Dict, Optional

from src.plugin_system.base.service_types import PluginServiceInfo
from src.plugin_system.core.plugin_service_registry import plugin_service_registry


def register_service(service_info: PluginServiceInfo, service_handler: Callable[..., Any]) -> bool:
    """注册插件服务。"""
    return plugin_service_registry.register_service(service_info, service_handler)


def get_service(service_name: str, plugin_name: Optional[str] = None) -> Optional[PluginServiceInfo]:
    """获取插件服务元信息。"""
    return plugin_service_registry.get_service(service_name, plugin_name)


def get_service_handler(service_name: str, plugin_name: Optional[str] = None) -> Optional[Callable[..., Any]]:
    """获取插件服务处理函数。"""
    return plugin_service_registry.get_service_handler(service_name, plugin_name)


def list_services(plugin_name: Optional[str] = None, enabled_only: bool = False) -> Dict[str, PluginServiceInfo]:
    """列出插件服务。"""
    return plugin_service_registry.list_services(plugin_name=plugin_name, enabled_only=enabled_only)


def enable_service(service_name: str, plugin_name: Optional[str] = None) -> bool:
    """启用插件服务。"""
    return plugin_service_registry.enable_service(service_name, plugin_name)


def disable_service(service_name: str, plugin_name: Optional[str] = None) -> bool:
    """禁用插件服务。"""
    return plugin_service_registry.disable_service(service_name, plugin_name)


def unregister_service(service_name: str, plugin_name: Optional[str] = None) -> bool:
    """注销插件服务。"""
    return plugin_service_registry.unregister_service(service_name, plugin_name)


async def call_service(service_name: str, *args: Any, plugin_name: Optional[str] = None, **kwargs: Any) -> Any:
    """调用插件服务。"""
    return await plugin_service_registry.call_service(service_name, *args, plugin_name=plugin_name, **kwargs)
