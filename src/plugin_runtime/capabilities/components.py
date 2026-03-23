from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Protocol

from src.common.logger import get_logger

logger = get_logger("plugin_runtime.integration")

if TYPE_CHECKING:
    from src.plugin_runtime.host.api_registry import APIEntry
    from src.plugin_runtime.host.component_registry import ComponentEntry
    from src.plugin_runtime.host.supervisor import PluginSupervisor


class _RuntimeComponentManagerProtocol(Protocol):
    @property
    def supervisors(self) -> List["PluginSupervisor"]: ...

    def _get_supervisor_for_plugin(self, plugin_id: str) -> Optional["PluginSupervisor"]: ...

    def _resolve_component_toggle_target(
        self, name: str, component_type: str
    ) -> tuple[Optional["ComponentEntry"], Optional[str]]: ...

    def _find_duplicate_plugin_ids(self, plugin_dirs: List[Path]) -> Dict[str, List[Path]]: ...

    def _iter_plugin_dirs(self) -> Iterable[Path]: ...


class RuntimeComponentCapabilityMixin:
    @staticmethod
    def _normalize_component_type(component_type: str) -> str:
        """规范化组件类型名称。

        Args:
            component_type: 原始组件类型。

        Returns:
            str: 统一转为大写后的组件类型名。
        """

        return str(component_type or "").strip().upper()

    @classmethod
    def _is_api_component_type(cls, component_type: str) -> bool:
        """判断组件类型是否为 API。

        Args:
            component_type: 原始组件类型。

        Returns:
            bool: 是否为 API 组件类型。
        """

        return cls._normalize_component_type(component_type) == "API"

    @staticmethod
    def _serialize_api_entry(entry: "APIEntry") -> Dict[str, Any]:
        """将 API 组件条目序列化为能力返回值。

        Args:
            entry: API 组件条目。

        Returns:
            Dict[str, Any]: 适合通过能力层返回给插件的 API 元信息。
        """

        return {
            "name": entry.name,
            "full_name": entry.full_name,
            "plugin_id": entry.plugin_id,
            "description": entry.description,
            "version": entry.version,
            "public": entry.public,
            "enabled": entry.enabled,
            "metadata": dict(entry.metadata),
        }

    @classmethod
    def _serialize_api_component_entry(cls, entry: "APIEntry") -> Dict[str, Any]:
        """将 API 条目序列化为通用组件视图。

        Args:
            entry: API 组件条目。

        Returns:
            Dict[str, Any]: 适合 ``component.get_all_plugins`` 返回的组件结构。
        """

        serialized_entry = cls._serialize_api_entry(entry)
        return {
            "name": serialized_entry["name"],
            "full_name": serialized_entry["full_name"],
            "type": "API",
            "enabled": serialized_entry["enabled"],
            "metadata": serialized_entry["metadata"],
        }

    @staticmethod
    def _is_api_visible_to_plugin(entry: "APIEntry", caller_plugin_id: str) -> bool:
        """判断某个 API 是否对调用方可见。

        Args:
            entry: 目标 API 组件条目。
            caller_plugin_id: 调用方插件 ID。

        Returns:
            bool: 是否允许当前插件可见并调用。
        """

        return entry.plugin_id == caller_plugin_id or entry.public

    def _resolve_api_target(
        self: _RuntimeComponentManagerProtocol,
        caller_plugin_id: str,
        api_name: str,
        version: str = "",
    ) -> tuple[Optional["PluginSupervisor"], Optional["APIEntry"], Optional[str]]:
        """解析 API 名称到唯一可调用的目标组件。

        Args:
            caller_plugin_id: 调用方插件 ID。
            api_name: API 名称，支持 ``plugin_id.api_name`` 或唯一短名。
            version: 可选的 API 版本。

        Returns:
            tuple[Optional[PluginSupervisor], Optional[APIEntry], Optional[str]]:
                解析成功时返回 ``(监督器, API 条目, None)``，失败时返回错误信息。
        """

        normalized_api_name = str(api_name or "").strip()
        normalized_version = str(version or "").strip()
        if not normalized_api_name:
            return None, None, "缺少必要参数 api_name"

        if "." in normalized_api_name:
            target_plugin_id, target_api_name = normalized_api_name.split(".", 1)
            try:
                supervisor = self._get_supervisor_for_plugin(target_plugin_id)
            except RuntimeError as exc:
                return None, None, str(exc)

            if supervisor is None:
                return None, None, f"未找到 API 提供方插件: {target_plugin_id}"

            entry = supervisor.api_registry.get_api(
                plugin_id=target_plugin_id,
                name=target_api_name,
                enabled_only=True,
            )
            if entry is None:
                return None, None, f"未找到 API: {normalized_api_name}"
            if normalized_version and entry.version != normalized_version:
                return None, None, f"未找到版本为 {normalized_version} 的 API: {normalized_api_name}"
            if not self._is_api_visible_to_plugin(entry, caller_plugin_id):
                return None, None, f"API {normalized_api_name} 未公开，禁止跨插件调用"
            return supervisor, entry, None

        visible_matches: List[tuple["PluginSupervisor", "APIEntry"]] = []
        hidden_match_exists = False
        for supervisor in self.supervisors:
            for entry in supervisor.api_registry.get_apis(name=normalized_api_name, enabled_only=True):
                if normalized_version and entry.version != normalized_version:
                    continue
                if self._is_api_visible_to_plugin(entry, caller_plugin_id):
                    visible_matches.append((supervisor, entry))
                else:
                    hidden_match_exists = True

        if len(visible_matches) == 1:
            return visible_matches[0][0], visible_matches[0][1], None
        if len(visible_matches) > 1:
            return None, None, f"API 名称不唯一: {normalized_api_name}，请使用 plugin_id.api_name"
        if hidden_match_exists:
            return None, None, f"API {normalized_api_name} 未公开，禁止跨插件调用"
        if normalized_version:
            return None, None, f"未找到版本为 {normalized_version} 的 API: {normalized_api_name}"
        return None, None, f"未找到 API: {normalized_api_name}"

    def _resolve_api_toggle_target(
        self: _RuntimeComponentManagerProtocol,
        name: str,
    ) -> tuple[Optional["PluginSupervisor"], Optional["APIEntry"], Optional[str]]:
        """解析需要启用或禁用的 API 组件。

        Args:
            name: API 名称，支持 ``plugin_id.api_name`` 或唯一短名。

        Returns:
            tuple[Optional[PluginSupervisor], Optional[APIEntry], Optional[str]]:
                解析成功时返回 ``(监督器, API 条目, None)``，失败时返回错误信息。
        """

        normalized_name = str(name or "").strip()
        if not normalized_name:
            return None, None, "缺少必要参数 name"

        if "." in normalized_name:
            plugin_id, api_name = normalized_name.split(".", 1)
            try:
                supervisor = self._get_supervisor_for_plugin(plugin_id)
            except RuntimeError as exc:
                return None, None, str(exc)

            if supervisor is None:
                return None, None, f"未找到 API 提供方插件: {plugin_id}"

            entry = supervisor.api_registry.get_api(
                plugin_id=plugin_id,
                name=api_name,
                enabled_only=False,
            )
            if entry is None:
                return None, None, f"未找到 API: {normalized_name}"
            return supervisor, entry, None

        matches: List[tuple["PluginSupervisor", "APIEntry"]] = []
        for supervisor in self.supervisors:
            for entry in supervisor.api_registry.get_apis(name=normalized_name, enabled_only=False):
                matches.append((supervisor, entry))

        if len(matches) == 1:
            return matches[0][0], matches[0][1], None
        if len(matches) > 1:
            return None, None, f"API 名称不唯一: {normalized_name}，请使用 plugin_id.api_name"
        return None, None, f"未找到 API: {normalized_name}"

    async def _cap_component_get_all_plugins(
        self: _RuntimeComponentManagerProtocol, plugin_id: str, capability: str, args: Dict[str, Any]
    ) -> Any:
        result: Dict[str, Any] = {}
        for sv in self.supervisors:
            for pid, reg in sv._registered_plugins.items():
                if pid in result:
                    logger.error(f"检测到重复插件 ID {pid}，component.get_all_plugins 结果已拒绝聚合")
                    return {"success": False, "error": f"检测到重复插件 ID: {pid}"}
                comps = sv.component_registry.get_components_by_plugin(pid, enabled_only=False)
                components_list = [
                    {
                        "name": component.name,
                        "full_name": component.full_name,
                        "type": component.component_type,
                        "enabled": component.enabled,
                        "metadata": component.metadata,
                    }
                    for component in comps
                ]
                components_list.extend(
                    self._serialize_api_component_entry(entry)
                    for entry in sv.api_registry.get_apis(plugin_id=pid, enabled_only=False)
                )
                result[pid] = {
                    "name": pid,
                    "version": reg.plugin_version,
                    "description": "",
                    "author": "",
                    "enabled": True,
                    "components": components_list,
                }
        return {"success": True, "plugins": result}

    async def _cap_component_get_plugin_info(
        self: _RuntimeComponentManagerProtocol, plugin_id: str, capability: str, args: Dict[str, Any]
    ) -> Any:
        plugin_name: str = args.get("plugin_name", plugin_id)
        try:
            sv = self._get_supervisor_for_plugin(plugin_name)
        except RuntimeError as exc:
            return {"success": False, "error": str(exc)}

        if sv is not None and (reg := sv._registered_plugins.get(plugin_name)) is not None:
            return {
                "success": True,
                "plugin": {
                    "name": plugin_name,
                    "version": reg.plugin_version,
                    "description": "",
                    "author": "",
                    "enabled": True,
                },
            }
        return {"success": False, "error": f"未找到插件: {plugin_name}"}

    async def _cap_component_list_loaded_plugins(
        self: _RuntimeComponentManagerProtocol, plugin_id: str, capability: str, args: Dict[str, Any]
    ) -> Any:
        plugins: List[str] = []
        for sv in self.supervisors:
            plugins.extend(sv._registered_plugins.keys())
        return {"success": True, "plugins": plugins}

    async def _cap_component_list_registered_plugins(
        self: _RuntimeComponentManagerProtocol, plugin_id: str, capability: str, args: Dict[str, Any]
    ) -> Any:
        plugins: List[str] = []
        for sv in self.supervisors:
            plugins.extend(sv._registered_plugins.keys())
        return {"success": True, "plugins": plugins}

    def _resolve_component_toggle_target(
        self: _RuntimeComponentManagerProtocol, name: str, component_type: str
    ) -> tuple[Optional["ComponentEntry"], Optional[str]]:
        normalized_component_type = self._normalize_component_type(component_type)
        short_name_matches: List["ComponentEntry"] = []
        for sv in self.supervisors:
            comp = sv.component_registry.get_component(name)
            if comp is not None and comp.component_type == normalized_component_type:
                return comp, None

            short_name_matches.extend(
                candidate
                for candidate in sv.component_registry.get_components_by_type(
                    normalized_component_type,
                    enabled_only=False,
                )
                if candidate.name == name
            )

        if len(short_name_matches) == 1:
            return short_name_matches[0], None
        if len(short_name_matches) > 1:
            return None, f"组件名不唯一: {name} ({normalized_component_type})，请使用完整名 plugin_id.component_name"
        return None, f"未找到组件: {name} ({normalized_component_type})"

    async def _cap_component_enable(
        self: _RuntimeComponentManagerProtocol, plugin_id: str, capability: str, args: Dict[str, Any]
    ) -> Any:
        name: str = args.get("name", "")
        component_type: str = args.get("component_type", "")
        scope: str = args.get("scope", "global")
        stream_id: str = args.get("stream_id", "")
        if not name or not component_type:
            return {"success": False, "error": "缺少必要参数 name 或 component_type"}
        if scope != "global" or stream_id:
            return {"success": False, "error": "当前仅支持全局组件启用，不支持 scope/stream_id 定位"}

        if self._is_api_component_type(component_type):
            supervisor, api_entry, error = self._resolve_api_toggle_target(name)
            if supervisor is None or api_entry is None:
                return {"success": False, "error": error or f"未找到 API: {name}"}
            supervisor.api_registry.toggle_api_status(api_entry.full_name, True)
            return {"success": True}

        comp, error = self._resolve_component_toggle_target(name, component_type)
        if comp is None:
            return {"success": False, "error": error or f"未找到组件: {name} ({component_type})"}

        comp.enabled = True
        return {"success": True}

    async def _cap_component_disable(
        self: _RuntimeComponentManagerProtocol, plugin_id: str, capability: str, args: Dict[str, Any]
    ) -> Any:
        name: str = args.get("name", "")
        component_type: str = args.get("component_type", "")
        scope: str = args.get("scope", "global")
        stream_id: str = args.get("stream_id", "")
        if not name or not component_type:
            return {"success": False, "error": "缺少必要参数 name 或 component_type"}
        if scope != "global" or stream_id:
            return {"success": False, "error": "当前仅支持全局组件禁用，不支持 scope/stream_id 定位"}

        if self._is_api_component_type(component_type):
            supervisor, api_entry, error = self._resolve_api_toggle_target(name)
            if supervisor is None or api_entry is None:
                return {"success": False, "error": error or f"未找到 API: {name}"}
            supervisor.api_registry.toggle_api_status(api_entry.full_name, False)
            return {"success": True}

        comp, error = self._resolve_component_toggle_target(name, component_type)
        if comp is None:
            return {"success": False, "error": error or f"未找到组件: {name} ({component_type})"}

        comp.enabled = False
        return {"success": True}

    async def _cap_component_load_plugin(
        self: _RuntimeComponentManagerProtocol, plugin_id: str, capability: str, args: Dict[str, Any]
    ) -> Any:
        plugin_name: str = args.get("plugin_name", "")
        if not plugin_name:
            return {"success": False, "error": "缺少必要参数 plugin_name"}

        if duplicate_plugin_ids := self._find_duplicate_plugin_ids(list(self._iter_plugin_dirs())):
            details = "; ".join(
                f"{conflict_plugin_id}: {', '.join(str(path) for path in paths)}"
                for conflict_plugin_id, paths in sorted(duplicate_plugin_ids.items())
            )
            return {"success": False, "error": f"检测到重复插件 ID，拒绝热重载: {details}"}

        try:
            registered_supervisor = self._get_supervisor_for_plugin(plugin_name)
        except RuntimeError as exc:
            return {"success": False, "error": str(exc)}

        if registered_supervisor is not None:
            try:
                reloaded = await registered_supervisor.reload_plugins(
                    plugin_ids=[plugin_name],
                    reason=f"load {plugin_name}",
                )
                if reloaded:
                    return {"success": True, "count": 1}
                return {"success": False, "error": f"插件 {plugin_name} 热重载失败，已回滚"}
            except Exception as e:
                logger.error(f"[cap.component.load_plugin] 热重载失败: {e}")
                return {"success": False, "error": str(e)}

        for sv in self.supervisors:
            for pdir in sv._plugin_dirs:
                if (pdir / plugin_name).is_dir():
                    try:
                        reloaded = await sv.reload_plugins(
                            plugin_ids=[plugin_name],
                            reason=f"load {plugin_name}",
                        )
                        if reloaded:
                            return {"success": True, "count": 1}
                        return {"success": False, "error": f"插件 {plugin_name} 热重载失败，已回滚"}
                    except Exception as e:
                        logger.error(f"[cap.component.load_plugin] 热重载失败: {e}")
                        return {"success": False, "error": str(e)}

        return {"success": False, "error": f"未找到插件: {plugin_name}"}

    async def _cap_component_unload_plugin(
        self: _RuntimeComponentManagerProtocol, plugin_id: str, capability: str, args: Dict[str, Any]
    ) -> Any:
        return {"success": False, "error": "新运行时不支持单独卸载插件，请使用 reload"}

    async def _cap_component_reload_plugin(
        self: _RuntimeComponentManagerProtocol, plugin_id: str, capability: str, args: Dict[str, Any]
    ) -> Any:
        plugin_name: str = args.get("plugin_name", "")
        if not plugin_name:
            return {"success": False, "error": "缺少必要参数 plugin_name"}

        if duplicate_plugin_ids := self._find_duplicate_plugin_ids(list(self._iter_plugin_dirs())):
            details = "; ".join(
                f"{conflict_plugin_id}: {', '.join(str(path) for path in paths)}"
                for conflict_plugin_id, paths in sorted(duplicate_plugin_ids.items())
            )
            return {"success": False, "error": f"检测到重复插件 ID，拒绝热重载: {details}"}

        try:
            sv = self._get_supervisor_for_plugin(plugin_name)
        except RuntimeError as exc:
            return {"success": False, "error": str(exc)}

        if sv is not None:
            try:
                reloaded = await sv.reload_plugins(
                    plugin_ids=[plugin_name],
                    reason=f"reload {plugin_name}",
                )
                if reloaded:
                    return {"success": True}
                return {"success": False, "error": f"插件 {plugin_name} 热重载失败，已回滚"}
            except Exception as e:
                logger.error(f"[cap.component.reload_plugin] 热重载失败: {e}")
                return {"success": False, "error": str(e)}
        return {"success": False, "error": f"未找到插件: {plugin_name}"}

    async def _cap_api_call(
        self: _RuntimeComponentManagerProtocol,
        plugin_id: str,
        capability: str,
        args: Dict[str, Any],
    ) -> Any:
        """调用其他插件公开的 API。

        Args:
            plugin_id: 当前调用方插件 ID。
            capability: 能力名称。
            args: 能力参数。

        Returns:
            Any: API 调用结果。
        """

        del capability
        api_name = str(args.get("api_name", "") or "").strip()
        version = str(args.get("version", "") or "").strip()
        api_args = args.get("args", {})
        if not isinstance(api_args, dict):
            return {"success": False, "error": "参数 args 必须为字典"}

        supervisor, entry, error = self._resolve_api_target(plugin_id, api_name, version)
        if supervisor is None or entry is None:
            return {"success": False, "error": error or "API 解析失败"}

        try:
            response = await supervisor.invoke_api(
                plugin_id=entry.plugin_id,
                component_name=entry.name,
                args=api_args,
                timeout_ms=30000,
            )
        except Exception as exc:
            logger.error(f"[cap.api.call] 调用 API {entry.full_name} 失败: {exc}", exc_info=True)
            return {"success": False, "error": str(exc)}

        if response.error:
            return {"success": False, "error": response.error.get("message", "API 调用失败")}

        payload = response.payload if isinstance(response.payload, dict) else {}
        if not bool(payload.get("success", False)):
            result = payload.get("result")
            return {"success": False, "error": "" if result is None else str(result)}
        return {"success": True, "result": payload.get("result")}

    async def _cap_api_get(
        self: _RuntimeComponentManagerProtocol,
        plugin_id: str,
        capability: str,
        args: Dict[str, Any],
    ) -> Any:
        """获取当前插件可见的单个 API 元信息。

        Args:
            plugin_id: 当前调用方插件 ID。
            capability: 能力名称。
            args: 能力参数。

        Returns:
            Any: API 元信息或 ``None``。
        """

        del capability
        api_name = str(args.get("api_name", "") or "").strip()
        version = str(args.get("version", "") or "").strip()
        if not api_name:
            return {"success": False, "error": "缺少必要参数 api_name"}

        supervisor, entry, _error = self._resolve_api_target(plugin_id, api_name, version)
        if supervisor is None or entry is None:
            return {"success": True, "api": None}
        return {"success": True, "api": self._serialize_api_entry(entry)}

    async def _cap_api_list(
        self: _RuntimeComponentManagerProtocol,
        plugin_id: str,
        capability: str,
        args: Dict[str, Any],
    ) -> Any:
        """列出当前插件可见的 API 列表。

        Args:
            plugin_id: 当前调用方插件 ID。
            capability: 能力名称。
            args: 能力参数。

        Returns:
            Any: API 元信息列表。
        """

        del capability
        target_plugin_id = str(args.get("plugin_id", "") or "").strip()
        apis: List[Dict[str, Any]] = []
        for supervisor in self.supervisors:
            for entry in supervisor.api_registry.get_apis(
                plugin_id=target_plugin_id or None,
                enabled_only=True,
            ):
                if not self._is_api_visible_to_plugin(entry, plugin_id):
                    continue
                apis.append(self._serialize_api_entry(entry))

        apis.sort(key=lambda item: (str(item["plugin_id"]), str(item["name"]), str(item["version"])))
        return {"success": True, "apis": apis}
