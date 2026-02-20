from typing import Any, Callable, Dict, Optional
import inspect

from src.common.logger import get_logger
from src.plugin_system.base.service_types import PluginServiceInfo

logger = get_logger("plugin_service_registry")


class PluginServiceRegistry:
    """插件服务注册中心"""

    def __init__(self):
        self._services: Dict[str, PluginServiceInfo] = {}
        self._service_handlers: Dict[str, Callable[..., Any]] = {}
        logger.info("插件服务注册中心初始化完成")

    def register_service(self, service_info: PluginServiceInfo, service_handler: Callable[..., Any]) -> bool:
        """注册插件服务。"""
        if not service_info.name or not service_info.plugin_name:
            logger.error("插件服务注册失败: service名称或插件名称为空")
            return False
        if "." in service_info.name:
            logger.error(f"插件服务名称 '{service_info.name}' 包含非法字符 '.'，请使用下划线替代")
            return False
        if "." in service_info.plugin_name:
            logger.error(f"插件服务所属插件名称 '{service_info.plugin_name}' 包含非法字符 '.'，请使用下划线替代")
            return False

        full_name = service_info.full_name
        if full_name in self._services:
            logger.warning(f"插件服务已存在，拒绝重复注册: {full_name}")
            return False

        self._services[full_name] = service_info
        self._service_handlers[full_name] = service_handler
        logger.debug(f"已注册插件服务: {full_name} (version={service_info.version})")
        return True

    def get_service(self, service_name: str, plugin_name: Optional[str] = None) -> Optional[PluginServiceInfo]:
        """获取插件服务元信息。

        service_name支持：
        - full_name: plugin_name.service_name
        - short_name: service_name（当唯一时可解析）
        """
        full_name = self._resolve_full_name(service_name, plugin_name)
        return self._services.get(full_name) if full_name else None

    def get_service_handler(self, service_name: str, plugin_name: Optional[str] = None) -> Optional[Callable[..., Any]]:
        """获取插件服务处理函数。"""
        full_name = self._resolve_full_name(service_name, plugin_name)
        return self._service_handlers.get(full_name) if full_name else None

    def list_services(self, plugin_name: Optional[str] = None, enabled_only: bool = False) -> Dict[str, PluginServiceInfo]:
        """列出插件服务。"""
        services = self._services.copy()
        if plugin_name:
            services = {name: info for name, info in services.items() if info.plugin_name == plugin_name}
        if enabled_only:
            services = {name: info for name, info in services.items() if info.enabled}
        return services

    def enable_service(self, service_name: str, plugin_name: Optional[str] = None) -> bool:
        """启用插件服务。"""
        if not (service_info := self.get_service(service_name, plugin_name)):
            logger.warning(f"插件服务未注册，无法启用: {service_name}")
            return False
        service_info.enabled = True
        logger.info(f"插件服务已启用: {service_info.full_name}")
        return True

    def disable_service(self, service_name: str, plugin_name: Optional[str] = None) -> bool:
        """禁用插件服务。"""
        if not (service_info := self.get_service(service_name, plugin_name)):
            logger.warning(f"插件服务未注册，无法禁用: {service_name}")
            return False
        service_info.enabled = False
        logger.info(f"插件服务已禁用: {service_info.full_name}")
        return True

    def unregister_service(self, service_name: str, plugin_name: Optional[str] = None) -> bool:
        """注销单个插件服务。"""
        full_name = self._resolve_full_name(service_name, plugin_name)
        if not full_name:
            logger.warning(f"插件服务未注册，无法注销: {service_name}")
            return False

        self._services.pop(full_name, None)
        self._service_handlers.pop(full_name, None)
        logger.info(f"插件服务已注销: {full_name}")
        return True

    def remove_services_by_plugin(self, plugin_name: str) -> int:
        """移除某插件的所有注册服务。"""
        target_names = [full_name for full_name, info in self._services.items() if info.plugin_name == plugin_name]
        for full_name in target_names:
            self._services.pop(full_name, None)
            self._service_handlers.pop(full_name, None)

        removed_count = len(target_names)
        if removed_count:
            logger.info(f"已移除插件 {plugin_name} 的服务数量: {removed_count}")
        return removed_count

    async def call_service(self, service_name: str, *args: Any, plugin_name: Optional[str] = None, **kwargs: Any) -> Any:
        """调用插件服务（支持同步/异步handler）。"""
        service_info = self.get_service(service_name, plugin_name)
        if not service_info:
            target_name = f"{plugin_name}.{service_name}" if plugin_name and "." not in service_name else service_name
            raise ValueError(f"插件服务未注册: {target_name}")
        if not service_info.enabled:
            raise RuntimeError(f"插件服务已禁用: {service_info.full_name}")

        handler = self.get_service_handler(service_name, plugin_name)
        if not handler:
            raise RuntimeError(f"插件服务处理器不存在: {service_info.full_name}")

        result = handler(*args, **kwargs)
        return await result if inspect.isawaitable(result) else result

    def _resolve_full_name(self, service_name: str, plugin_name: Optional[str] = None) -> Optional[str]:
        """解析服务全名。"""
        if "." in service_name:
            return service_name if service_name in self._services else None

        if plugin_name:
            full_name = f"{plugin_name}.{service_name}"
            return full_name if full_name in self._services else None

        candidates = [full_name for full_name, info in self._services.items() if info.name == service_name]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            logger.warning(f"插件服务名称 '{service_name}' 存在多义，请传入plugin_name或使用完整服务名")
            return None
        return None


plugin_service_registry = PluginServiceRegistry()
