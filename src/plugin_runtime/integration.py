"""插件运行时与主程序的集成层

提供 PluginRuntimeManager 单例，负责：
1. 管理双 PluginSupervisor 的生命周期（内置插件 / 第三方插件各一个子进程）
2. 将 EventType 桥接到运行时的 event dispatch
3. 在运行时的 ComponentRegistry 中查找命令
4. 提供统一的能力实现注册接口，使插件可以调用主程序功能
"""

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

import asyncio
import json
import os
from pathlib import Path

from src.common.logger import get_logger
from src.config.config import config_manager, global_config
from src.config.file_watcher import FileChange, FileWatcher
from src.plugin_runtime.capabilities import (
    RuntimeComponentCapabilityMixin,
    RuntimeCoreCapabilityMixin,
    RuntimeDataCapabilityMixin,
)
from src.plugin_runtime.capabilities.registry import register_capability_impls

if TYPE_CHECKING:
    from src.plugin_runtime.host.supervisor import PluginSupervisor

logger = get_logger("plugin_runtime.integration")

# 旧系统 EventType -> 新系统 event_type 字符串映射
_EVENT_TYPE_MAP: Dict[str, str] = {
    "on_start": "on_start",
    "on_stop": "on_stop",
    "on_message_pre_process": "on_message_pre_process",
    "on_message": "on_message",
    "on_plan": "on_plan",
    "post_llm": "post_llm",
    "after_llm": "after_llm",
    "post_send_pre_process": "post_send_pre_process",
    "post_send": "post_send",
    "after_send": "after_send",
}


class PluginRuntimeManager(
    RuntimeCoreCapabilityMixin,
    RuntimeDataCapabilityMixin,
    RuntimeComponentCapabilityMixin,
):
    """插件运行时管理器（单例）

    内置插件与第三方插件分别运行在各自的 Supervisor / Runner 子进程中。
    """

    def __init__(self) -> None:
        from src.plugin_runtime.host.supervisor import PluginSupervisor

        self._builtin_supervisor: Optional[PluginSupervisor] = None
        self._thirdparty_supervisor: Optional[PluginSupervisor] = None
        self._started: bool = False
        self._config_reload_callback_registered: bool = False
        self._plugin_file_watcher: Optional[FileWatcher] = None
        self._plugin_file_watcher_subscription_id: Optional[str] = None

    # ─── 插件目录 ─────────────────────────────────────────────

    @staticmethod
    def _get_builtin_plugin_dirs() -> List[str]:
        """内置插件目录: src/plugins/built_in/"""
        candidate = os.path.abspath(os.path.join("src", "plugins", "built_in"))
        return [candidate] if os.path.isdir(candidate) else []

    @staticmethod
    def _get_thirdparty_plugin_dirs() -> List[str]:
        """第三方插件目录: plugins/"""
        candidate = os.path.abspath("plugins")
        return [candidate] if os.path.isdir(candidate) else []

    # ─── 生命周期 ─────────────────────────────────────────────

    async def start(self) -> None:
        """启动双子进程插件运行时"""
        if self._started:
            logger.warning("PluginRuntimeManager 已在运行中，跳过重复启动")
            return

        _cfg = global_config.plugin_runtime
        if not _cfg.enabled:
            logger.info("插件运行时已在配置中禁用，跳过启动")
            return

        from src.plugin_runtime.host.supervisor import PluginSupervisor

        builtin_dirs = self._get_builtin_plugin_dirs()
        thirdparty_dirs = self._get_thirdparty_plugin_dirs()

        if duplicate_plugin_ids := self._find_duplicate_plugin_ids(builtin_dirs + thirdparty_dirs):
            details = "; ".join(
                f"{plugin_id}: {', '.join(paths)}" for plugin_id, paths in sorted(duplicate_plugin_ids.items())
            )
            logger.error(f"检测到重复插件 ID，拒绝启动插件运行时: {details}")
            return

        if not builtin_dirs and not thirdparty_dirs:
            logger.info("未找到任何插件目录，跳过插件运行时启动")
            return

        # 从配置读取自定义 IPC socket 路径（留空则自动生成）
        socket_path_base = _cfg.ipc_socket_path or None

        # 当用户指定了自定义路径时，为两个 Supervisor 添加后缀以避免 UDS 冲突
        builtin_socket = f"{socket_path_base}-builtin" if socket_path_base else None
        thirdparty_socket = f"{socket_path_base}-thirdparty" if socket_path_base else None

        # 创建两个 Supervisor，各自拥有独立的 socket / Runner 子进程
        if builtin_dirs:
            self._builtin_supervisor = PluginSupervisor(
                plugin_dirs=builtin_dirs,
                socket_path=builtin_socket,
            )
            self._register_capability_impls(self._builtin_supervisor)

        if thirdparty_dirs:
            self._thirdparty_supervisor = PluginSupervisor(
                plugin_dirs=thirdparty_dirs,
                socket_path=thirdparty_socket,
            )
            self._register_capability_impls(self._thirdparty_supervisor)

        started_supervisors = []
        try:
            if self._builtin_supervisor:
                await self._builtin_supervisor.start()
                started_supervisors.append(self._builtin_supervisor)
            if self._thirdparty_supervisor:
                await self._thirdparty_supervisor.start()
                started_supervisors.append(self._thirdparty_supervisor)
            self._register_config_reload_callback()
            await self._start_plugin_file_watcher()
            self._started = True
            logger.info(f"插件运行时已启动 — 内置: {builtin_dirs or '无'}, 第三方: {thirdparty_dirs or '无'}")
        except Exception as e:
            logger.error(f"插件运行时启动失败: {e}", exc_info=True)
            await self._stop_plugin_file_watcher()
            self._unregister_config_reload_callback()
            await asyncio.gather(*(sv.stop() for sv in started_supervisors), return_exceptions=True)
            self._started = False
            self._builtin_supervisor = None
            self._thirdparty_supervisor = None

    async def stop(self) -> None:
        """停止所有插件运行时"""
        if not self._started:
            return

        await self._stop_plugin_file_watcher()
        self._unregister_config_reload_callback()

        coros = []
        if self._builtin_supervisor:
            coros.append(self._builtin_supervisor.stop())
        if self._thirdparty_supervisor:
            coros.append(self._thirdparty_supervisor.stop())

        try:
            await asyncio.gather(*coros, return_exceptions=True)
            logger.info("插件运行时已停止")
        except Exception as e:
            logger.error(f"插件运行时停止失败: {e}", exc_info=True)
        finally:
            self._started = False
            self._builtin_supervisor = None
            self._thirdparty_supervisor = None

    @property
    def is_running(self) -> bool:
        return self._started

    @property
    def supervisors(self) -> List[Any]:
        """获取所有活跃的 Supervisor"""
        return [s for s in (self._builtin_supervisor, self._thirdparty_supervisor) if s is not None]

    async def notify_plugin_config_updated(
        self,
        plugin_id: str,
        config_data: Optional[Dict[str, Any]] = None,
        config_version: str = "",
    ) -> bool:
        """向拥有该插件的 Supervisor 推送配置更新事件。"""
        if not self._started:
            return False

        try:
            sv = self._get_supervisor_for_plugin(plugin_id)
        except RuntimeError as exc:
            logger.error(f"推送插件配置更新失败: {exc}")
            return False

        if sv is None:
            return False

        config_payload = (
            config_data
            if config_data is not None
            else self._load_plugin_config_for_supervisor(plugin_id, getattr(sv, "_plugin_dirs", []))
        )
        await sv.notify_plugin_config_updated(
            plugin_id=plugin_id,
            config_data=config_payload,
            config_version=config_version,
        )
        return True

    async def handle_config_reload(self) -> None:
        """处理主配置热重载后的插件配置通知。"""
        if not self._started:
            return

        tasks = [
            self.notify_plugin_config_updated(plugin_id)
            for sv in self.supervisors
            for plugin_id in list(sv._registered_plugins.keys())
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # ─── 事件桥接 ──────────────────────────────────────────────

    async def bridge_event(
        self,
        event_type_value: str,
        message_dict: Optional[Dict[str, Any]] = None,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """将事件分发到所有 Supervisor

        Returns:
            (continue_flag, modified_message_dict)
        """
        if not self._started:
            return True, None

        new_event_type: str = _EVENT_TYPE_MAP.get(event_type_value, event_type_value)
        modified: Optional[Dict[str, Any]] = None

        for sv in self.supervisors:
            try:
                cont, mod = await sv.dispatch_event(
                    event_type=new_event_type,
                    message=modified or message_dict,
                    extra_args=extra_args,
                )
                if mod is not None:
                    modified = mod
                if not cont:
                    return False, modified
            except Exception as e:
                logger.error(f"事件 {new_event_type} 分发失败: {e}", exc_info=True)

        return True, modified

    # ─── 命令查找 ──────────────────────────────────────────────

    def find_command_by_text(self, text: str) -> Optional[Dict[str, Any]]:
        """在所有 Supervisor 的 ComponentRegistry 中查找命令"""
        if not self._started:
            return None

        for sv in self.supervisors:
            match_result = sv.component_registry.find_command_by_text(text)
            if match_result is not None:
                comp, matched_groups = match_result
                return {
                    "name": comp.name,
                    "full_name": comp.full_name,
                    "component_type": comp.component_type,
                    "plugin_id": comp.plugin_id,
                    "metadata": comp.metadata,
                    "enabled": comp.enabled,
                    "matched_groups": matched_groups,
                }
        return None

    async def invoke_plugin(
        self,
        method: str,
        plugin_id: str,
        component_name: str,
        args: Optional[Dict[str, Any]] = None,
        timeout_ms: int = 30000,
    ) -> Any:
        """将插件调用路由到拥有该插件的 Supervisor"""
        sv = self._get_supervisor_for_plugin(plugin_id)
        if sv is None:
            raise RuntimeError(f"插件 {plugin_id} 未在任何 Supervisor 中注册")
        return await sv.invoke_plugin(
            method=method,
            plugin_id=plugin_id,
            component_name=component_name,
            args=args,
            timeout_ms=timeout_ms,
        )

    def _get_supervisors_for_plugin(self, plugin_id: str) -> List[Any]:
        return [
            supervisor
            for supervisor in self.supervisors
            if plugin_id in getattr(supervisor, "_registered_plugins", {})
        ]

    def _get_supervisor_for_plugin(self, plugin_id: str) -> Optional[Any]:
        matches = self._get_supervisors_for_plugin(plugin_id)
        if len(matches) > 1:
            raise RuntimeError(f"插件 {plugin_id} 同时存在于多个 Supervisor 中，无法安全路由")
        return matches[0] if matches else None

    @staticmethod
    def _find_duplicate_plugin_ids(plugin_dirs: List[str]) -> Dict[str, List[str]]:
        plugin_locations: Dict[str, List[str]] = {}
        for base_dir in plugin_dirs:
            if not os.path.isdir(base_dir):
                continue
            for entry in os.listdir(base_dir):
                plugin_dir = os.path.join(base_dir, entry)
                if not os.path.isdir(plugin_dir):
                    continue
                manifest_path = os.path.join(plugin_dir, "_manifest.json")
                plugin_path = os.path.join(plugin_dir, "plugin.py")
                if not os.path.exists(manifest_path) or not os.path.exists(plugin_path):
                    continue

                plugin_id = entry
                try:
                    with open(manifest_path, "r", encoding="utf-8") as manifest_file:
                        manifest = json.load(manifest_file)
                    plugin_id = str(manifest.get("name", entry)).strip() or entry
                except Exception:
                    continue

                plugin_locations.setdefault(plugin_id, []).append(plugin_dir)

        return {
            plugin_id: sorted(dict.fromkeys(paths))
            for plugin_id, paths in plugin_locations.items()
            if len(set(paths)) > 1
        }

    def _register_config_reload_callback(self) -> None:
        if self._config_reload_callback_registered:
            return
        config_manager.register_reload_callback(self.handle_config_reload)
        self._config_reload_callback_registered = True

    def _unregister_config_reload_callback(self) -> None:
        if not self._config_reload_callback_registered:
            return
        config_manager.unregister_reload_callback(self.handle_config_reload)
        self._config_reload_callback_registered = False

    async def _start_plugin_file_watcher(self) -> None:
        if self._plugin_file_watcher is not None and self._plugin_file_watcher.running:
            return

        watch_paths = [Path(path).resolve() for path in self._iter_plugin_dirs() if os.path.isdir(path)]
        if not watch_paths:
            return

        watcher = FileWatcher(
            paths=watch_paths,
            debounce_ms=600,
            callback_timeout_s=15.0,
            callback_failure_threshold=3,
            callback_cooldown_s=30.0,
        )
        subscription_id = watcher.subscribe(self._handle_plugin_file_changes, paths=watch_paths)
        await watcher.start()
        self._plugin_file_watcher = watcher
        self._plugin_file_watcher_subscription_id = subscription_id

    async def _stop_plugin_file_watcher(self) -> None:
        if self._plugin_file_watcher is None:
            return
        if self._plugin_file_watcher_subscription_id is not None:
            self._plugin_file_watcher.unsubscribe(self._plugin_file_watcher_subscription_id)
            self._plugin_file_watcher_subscription_id = None
        await self._plugin_file_watcher.stop()
        self._plugin_file_watcher = None

    def _iter_plugin_dirs(self) -> Iterable[str]:
        for supervisor in self.supervisors:
            yield from getattr(supervisor, "_plugin_dirs", [])

    async def _handle_plugin_file_changes(self, changes: List[FileChange]) -> None:
        if not self._started or not changes:
            return

        if duplicate_plugin_ids := self._find_duplicate_plugin_ids(list(self._iter_plugin_dirs())):
            details = "; ".join(
                f"{plugin_id}: {', '.join(paths)}" for plugin_id, paths in sorted(duplicate_plugin_ids.items())
            )
            logger.error(f"检测到重复插件 ID，跳过本次插件热重载: {details}")
            return

        reload_supervisors: List[Any] = []
        config_updates: Dict[str, set[str]] = {}
        changed_paths = [change.path.resolve() for change in changes]

        for supervisor in self.supervisors:
            plugin_ids_for_config = config_updates.setdefault(self._get_supervisor_key(supervisor), set())
            for path in changed_paths:
                plugin_id = self._match_plugin_id_for_supervisor(supervisor, path)
                if plugin_id is None:
                    continue
                if path.name == "config.toml":
                    plugin_ids_for_config.add(plugin_id)
                elif path.name in {"plugin.py", "_manifest.json"} or path.suffix == ".py":
                    if supervisor not in reload_supervisors:
                        reload_supervisors.append(supervisor)

        for supervisor in reload_supervisors:
            await supervisor.reload_plugins(reason="file_watcher")

        for supervisor in self.supervisors:
            if supervisor in reload_supervisors:
                continue
            for plugin_id in config_updates.get(self._get_supervisor_key(supervisor), set()):
                try:
                    await supervisor.notify_plugin_config_updated(
                        plugin_id=plugin_id,
                        config_data=self._load_plugin_config_for_supervisor(
                            plugin_id, getattr(supervisor, "_plugin_dirs", [])
                        ),
                    )
                except Exception as exc:
                    logger.warning(f"插件 {plugin_id} 配置热更新通知失败: {exc}")

    @staticmethod
    def _get_supervisor_key(supervisor: Any) -> int:
        return id(supervisor)

    @staticmethod
    def _plugin_dir_matches(path: Path, plugin_dir: str) -> bool:
        plugin_root = Path(plugin_dir).resolve()
        return path == plugin_root or path.is_relative_to(plugin_root)

    def _match_plugin_id_for_supervisor(self, supervisor: Any, path: Path) -> Optional[str]:
        for plugin_id, _reg in getattr(supervisor, "_registered_plugins", {}).items():
            for plugin_dir in getattr(supervisor, "_plugin_dirs", []):
                candidate_dir = Path(plugin_dir).resolve() / plugin_id
                if path == candidate_dir or path.is_relative_to(candidate_dir):
                    return plugin_id

        for plugin_dir in getattr(supervisor, "_plugin_dirs", []):
            plugin_root = Path(plugin_dir).resolve()
            if self._plugin_dir_matches(path, plugin_dir) and (relative_parts := path.relative_to(plugin_root).parts):
                return relative_parts[0]
        return None

    @staticmethod
    def _load_plugin_config_for_supervisor(plugin_id: str, plugin_dirs: Iterable[str]) -> Dict[str, Any]:
        import tomlkit

        for plugin_dir in plugin_dirs:
            plugin_path = Path(plugin_dir).resolve() / plugin_id
            if plugin_path.is_dir():
                config_path = plugin_path / "config.toml"
                if not config_path.exists():
                    return {}
                with open(config_path, "r", encoding="utf-8") as handle:
                    return dict(tomlkit.load(handle))
        return {}

    # ─── 能力实现注册 ──────────────────────────────────────────

    def _register_capability_impls(self, supervisor: "PluginSupervisor") -> None:
        register_capability_impls(self, supervisor)


# ─── 单例 ──────────────────────────────────────────────────

_manager: Optional[PluginRuntimeManager] = None


def get_plugin_runtime_manager() -> PluginRuntimeManager:
    """获取 PluginRuntimeManager 全局单例"""
    global _manager
    if _manager is None:
        _manager = PluginRuntimeManager()
    return _manager
