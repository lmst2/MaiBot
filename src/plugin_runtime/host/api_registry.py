"""Host 侧插件 API 动态注册表。"""

from typing import Any, Dict, List, Optional, Set

from src.common.logger import get_logger

logger = get_logger("plugin_runtime.host.api_registry")


class APIEntry:
    """API 组件条目。"""

    __slots__ = (
        "description",
        "disabled_session",
        "enabled",
        "full_name",
        "metadata",
        "name",
        "plugin_id",
        "public",
        "version",
    )

    def __init__(self, name: str, plugin_id: str, metadata: Dict[str, Any]) -> None:
        """初始化 API 组件条目。

        Args:
            name: API 名称。
            plugin_id: 所属插件 ID。
            metadata: API 元数据。
        """

        self.name: str = name
        self.full_name: str = f"{plugin_id}.{name}"
        self.plugin_id: str = plugin_id
        self.description: str = str(metadata.get("description", "") or "")
        self.version: str = str(metadata.get("version", "1") or "1").strip() or "1"
        self.public: bool = bool(metadata.get("public", False))
        self.metadata: Dict[str, Any] = dict(metadata)
        self.enabled: bool = bool(metadata.get("enabled", True))
        self.disabled_session: Set[str] = set()


class APIRegistry:
    """Host 侧插件 API 动态注册表。

    该注册表不直接面向 Runner，而是复用插件组件注册/卸载事件，
    维护面向 API 调用场景的专用索引。
    """

    def __init__(self) -> None:
        """初始化 API 注册表。"""

        self._apis: Dict[str, APIEntry] = {}
        self._by_plugin: Dict[str, List[APIEntry]] = {}
        self._by_name: Dict[str, List[APIEntry]] = {}

    def clear(self) -> None:
        """清空全部 API 注册状态。"""

        self._apis.clear()
        self._by_plugin.clear()
        self._by_name.clear()

    @staticmethod
    def _is_api_component(component_type: Any) -> bool:
        """判断组件声明是否属于 API。

        Args:
            component_type: 原始组件类型值。

        Returns:
            bool: 是否为 API 组件。
        """

        return str(component_type or "").strip().upper() == "API"

    @staticmethod
    def check_api_enabled(entry: APIEntry, session_id: Optional[str] = None) -> bool:
        """判断 API 条目当前是否处于启用状态。

        Args:
            entry: 待检查的 API 条目。
            session_id: 可选的会话 ID。

        Returns:
            bool: 当前是否可用。
        """

        if session_id and session_id in entry.disabled_session:
            return False
        return entry.enabled

    def register_api(self, name: str, plugin_id: str, metadata: Dict[str, Any]) -> bool:
        """注册单个 API 条目。

        Args:
            name: API 名称。
            plugin_id: 所属插件 ID。
            metadata: API 元数据。

        Returns:
            bool: 是否成功注册。
        """

        normalized_name = str(name or "").strip()
        if not normalized_name:
            logger.warning(f"插件 {plugin_id} 存在空 API 名称声明，已忽略")
            return False

        entry = APIEntry(name=normalized_name, plugin_id=plugin_id, metadata=metadata)
        if entry.full_name in self._apis:
            logger.warning(f"API {entry.full_name} 已存在，覆盖旧条目")
            self._remove_entry(self._apis[entry.full_name])

        self._apis[entry.full_name] = entry
        self._by_plugin.setdefault(plugin_id, []).append(entry)
        self._by_name.setdefault(entry.name, []).append(entry)
        return True

    def register_plugin_apis(self, plugin_id: str, components: List[Dict[str, Any]]) -> int:
        """批量注册某个插件声明的全部 API。

        Args:
            plugin_id: 插件 ID。
            components: 插件组件声明列表。

        Returns:
            int: 成功注册的 API 数量。
        """

        count = 0
        for component in components:
            if not self._is_api_component(component.get("component_type")):
                continue
            if self.register_api(
                name=str(component.get("name", "") or ""),
                plugin_id=plugin_id,
                metadata=component.get("metadata", {}) if isinstance(component.get("metadata"), dict) else {},
            ):
                count += 1
        return count

    def _remove_entry(self, entry: APIEntry) -> None:
        """从全部索引中移除单个 API 条目。

        Args:
            entry: 待移除的 API 条目。
        """

        self._apis.pop(entry.full_name, None)
        plugin_entries = self._by_plugin.get(entry.plugin_id)
        if plugin_entries is not None:
            self._by_plugin[entry.plugin_id] = [candidate for candidate in plugin_entries if candidate is not entry]
            if not self._by_plugin[entry.plugin_id]:
                self._by_plugin.pop(entry.plugin_id, None)

        name_entries = self._by_name.get(entry.name)
        if name_entries is not None:
            self._by_name[entry.name] = [candidate for candidate in name_entries if candidate is not entry]
            if not self._by_name[entry.name]:
                self._by_name.pop(entry.name, None)

    def remove_apis_by_plugin(self, plugin_id: str) -> int:
        """移除某个插件的全部 API。

        Args:
            plugin_id: 目标插件 ID。

        Returns:
            int: 被移除的 API 数量。
        """

        entries = list(self._by_plugin.get(plugin_id, []))
        for entry in entries:
            self._remove_entry(entry)
        return len(entries)

    def get_api_by_full_name(
        self,
        full_name: str,
        *,
        enabled_only: bool = True,
        session_id: Optional[str] = None,
    ) -> Optional[APIEntry]:
        """按完整名查询单个 API。

        Args:
            full_name: API 完整名，格式为 ``plugin_id.api_name``。
            enabled_only: 是否仅返回启用状态的 API。
            session_id: 可选的会话 ID。

        Returns:
            Optional[APIEntry]: 命中时返回 API 条目。
        """

        entry = self._apis.get(full_name)
        if entry is None:
            return None
        if enabled_only and not self.check_api_enabled(entry, session_id):
            return None
        return entry

    def get_api(
        self,
        plugin_id: str,
        name: str,
        *,
        enabled_only: bool = True,
        session_id: Optional[str] = None,
    ) -> Optional[APIEntry]:
        """按插件 ID 和短名查询单个 API。

        Args:
            plugin_id: 提供方插件 ID。
            name: API 短名。
            enabled_only: 是否仅返回启用状态的 API。
            session_id: 可选的会话 ID。

        Returns:
            Optional[APIEntry]: 命中时返回 API 条目。
        """

        return self.get_api_by_full_name(
            f"{plugin_id}.{name}",
            enabled_only=enabled_only,
            session_id=session_id,
        )

    def get_apis(
        self,
        *,
        plugin_id: Optional[str] = None,
        name: str = "",
        enabled_only: bool = True,
        session_id: Optional[str] = None,
    ) -> List[APIEntry]:
        """查询 API 列表。

        Args:
            plugin_id: 可选的插件 ID 过滤条件。
            name: 可选的 API 名称过滤条件。
            enabled_only: 是否仅返回启用状态的 API。
            session_id: 可选的会话 ID。

        Returns:
            List[APIEntry]: 符合条件的 API 条目列表。
        """

        normalized_name = str(name or "").strip()
        if plugin_id:
            candidates = list(self._by_plugin.get(plugin_id, []))
        elif normalized_name:
            candidates = list(self._by_name.get(normalized_name, []))
        else:
            candidates = list(self._apis.values())

        filtered_entries: List[APIEntry] = []
        for entry in candidates:
            if normalized_name and entry.name != normalized_name:
                continue
            if enabled_only and not self.check_api_enabled(entry, session_id):
                continue
            filtered_entries.append(entry)
        return filtered_entries

    def toggle_api_status(self, full_name: str, enabled: bool, session_id: Optional[str] = None) -> bool:
        """设置指定 API 的启用状态。

        Args:
            full_name: API 完整名。
            enabled: 目标启用状态。
            session_id: 可选的会话 ID，仅对该会话生效。

        Returns:
            bool: 是否设置成功。
        """

        entry = self._apis.get(full_name)
        if entry is None:
            return False
        if session_id:
            if enabled:
                entry.disabled_session.discard(session_id)
            else:
                entry.disabled_session.add(session_id)
        else:
            entry.enabled = enabled
        return True
