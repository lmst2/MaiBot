"""插件加载器

在 Runner 进程中负责发现和加载插件。
插件通过 SDK 编写，不再 import src.*。
支持：manifest 校验、依赖解析（拓扑排序）、生命周期钩子。
兼容旧版 src.plugin_system 插件（通过导入钩子 + LegacyPluginAdapter）。
"""

from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import importlib
import importlib.util
import json
import os
import sys

from src.common.logger import get_logger
from src.plugin_runtime.runner.manifest_validator import ManifestValidator

logger = get_logger("plugin_runtime.runner.plugin_loader")


class PluginMeta:
    """加载后的插件元数据"""

    def __init__(
        self,
        plugin_id: str,
        plugin_dir: str,
        plugin_instance: Any,
        manifest: Dict[str, Any],
    ) -> None:
        self.plugin_id = plugin_id
        self.plugin_dir = plugin_dir
        self.instance = plugin_instance
        self.manifest = manifest
        self.version = manifest.get("version", "1.0.0")
        self.capabilities_required = manifest.get("capabilities", [])
        self.dependencies: List[str] = self._extract_dependencies(manifest)

    @staticmethod
    def _extract_dependencies(manifest: Dict[str, Any]) -> List[str]:
        raw = manifest.get("dependencies", [])
        result: List[str] = []
        for dep in raw:
            if isinstance(dep, str):
                result.append(dep.strip())
            elif isinstance(dep, dict):
                if name := str(dep.get("name", "")).strip():
                    result.append(name)
        return result


class PluginLoader:
    """插件加载器

    扫描插件目录，加载符合 SDK 规范的插件。
    每个插件目录须包含：
    - _manifest.json: 插件元数据
    - plugin.py: 插件入口模块（导出 create_plugin 工厂函数）
    """

    def __init__(self, host_version: str = "") -> None:
        self._loaded_plugins: Dict[str, PluginMeta] = {}
        self._failed_plugins: Dict[str, str] = {}
        self._manifest_validator = ManifestValidator(host_version=host_version)
        self._compat_hook_installed = False

    def discover_and_load(self, plugin_dirs: List[str]) -> List[PluginMeta]:
        """扫描多个目录并加载所有插件（含依赖排序和 manifest 校验）

        Args:
            plugin_dirs: 插件目录列表

        Returns:
            成功加载的插件元数据列表（按依赖顺序）
        """
        # 第一阶段：发现并校验 manifest
        candidates: Dict[str, Tuple[str, Dict[str, Any], str]] = {}  # id -> (dir, manifest, plugin_path)
        duplicate_candidates: Dict[str, List[str]] = {}
        for base_dir in plugin_dirs:
            if not os.path.isdir(base_dir):
                logger.warning(f"插件目录不存在: {base_dir}")
                continue

            for entry in os.listdir(base_dir):
                plugin_dir = os.path.join(base_dir, entry)
                if not os.path.isdir(plugin_dir):
                    continue

                manifest_path = os.path.join(plugin_dir, "_manifest.json")
                plugin_path = os.path.join(plugin_dir, "plugin.py")

                if not os.path.exists(manifest_path) or not os.path.exists(plugin_path):
                    continue

                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                except Exception as e:
                    self._failed_plugins[entry] = f"manifest 解析失败: {e}"
                    logger.error(f"插件 {entry} manifest 解析失败: {e}")
                    continue

                if not self._manifest_validator.validate(manifest):
                    errors = "; ".join(self._manifest_validator.errors)
                    self._failed_plugins[entry] = f"manifest 校验失败: {errors}"
                    continue

                plugin_id = str(manifest.get("name", entry)).strip() or entry
                if plugin_id in duplicate_candidates:
                    duplicate_candidates[plugin_id].append(plugin_dir)
                    continue

                previous = candidates.get(plugin_id)
                if previous is not None:
                    duplicate_candidates[plugin_id] = [previous[0], plugin_dir]
                    candidates.pop(plugin_id, None)
                    continue

                candidates[plugin_id] = (plugin_dir, manifest, plugin_path)

        for plugin_id, conflict_dirs in duplicate_candidates.items():
            unique_dirs = sorted(dict.fromkeys(conflict_dirs))
            reason = f"检测到重复插件 ID: {plugin_id} -> {', '.join(unique_dirs)}"
            self._failed_plugins[plugin_id] = reason
            logger.error(reason)

        # 第二阶段：依赖解析（拓扑排序）
        load_order, failed_deps = self._resolve_dependencies(candidates)

        for pid, reason in failed_deps.items():
            self._failed_plugins[pid] = reason
            logger.error(f"插件 {pid} 依赖解析失败: {reason}")

        # 第三阶段：按依赖顺序加载
        results = []
        for plugin_id in load_order:
            plugin_dir, manifest, plugin_path = candidates[plugin_id]
            try:
                if meta := self._load_single_plugin(plugin_id, plugin_dir, manifest, plugin_path):
                    self._loaded_plugins[meta.plugin_id] = meta
                    results.append(meta)
            except Exception as e:
                self._failed_plugins[plugin_id] = str(e)
                logger.error(f"加载插件失败 [{plugin_id}]: {e}", exc_info=True)

        return results

    def get_plugin(self, plugin_id: str) -> Optional[PluginMeta]:
        """获取已加载的插件"""
        return self._loaded_plugins.get(plugin_id)

    def list_plugins(self) -> List[str]:
        """列出所有已加载的插件 ID"""
        return list(self._loaded_plugins.keys())

    @property
    def failed_plugins(self) -> Dict[str, str]:
        return dict(self._failed_plugins)

    # ──── 依赖解析 ────────────────────────────────────────────

    def _resolve_dependencies(
        self,
        candidates: Dict[str, Tuple[str, Dict[str, Any], str]],
    ) -> Tuple[List[str], Dict[str, str]]:
        """拓扑排序解析加载顺序，返回 (有序列表, 失败项 {id: reason})。"""
        available = set(candidates.keys())
        dep_graph: Dict[str, Set[str]] = {}
        failed: Dict[str, str] = {}

        for pid, (_, manifest, _) in candidates.items():
            raw_deps = manifest.get("dependencies", [])
            resolved: Set[str] = set()
            missing: List[str] = []
            for dep in raw_deps:
                dep_name = dep if isinstance(dep, str) else str(dep.get("name", ""))
                dep_name = dep_name.strip()
                if not dep_name or dep_name == pid:
                    continue
                if dep_name in available:
                    resolved.add(dep_name)
                else:
                    missing.append(dep_name)
            if missing:
                failed[pid] = f"缺少依赖: {', '.join(missing)}"
            dep_graph[pid] = resolved

        # 移除失败项
        for pid in failed:
            dep_graph.pop(pid, None)

        # Kahn 拓扑排序
        indegree = {pid: len(deps) for pid, deps in dep_graph.items()}
        reverse: Dict[str, Set[str]] = {pid: set() for pid in dep_graph}
        for pid, deps in dep_graph.items():
            for d in deps:
                if d in reverse:
                    reverse[d].add(pid)

        queue = deque(sorted(pid for pid, deg in indegree.items() if deg == 0))
        sorted_order: List[str] = []

        while queue:
            current = queue.popleft()
            sorted_order.append(current)
            for dependent in sorted(reverse.get(current, [])):
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    queue.append(dependent)

        cycle_plugins = {pid for pid, deg in indegree.items() if deg > 0}
        for pid in cycle_plugins:
            failed[pid] = "检测到循环依赖"

        return sorted_order, failed

    # ──── 单个插件加载 ────────────────────────────────────────

    def _load_single_plugin(
        self,
        plugin_id: str,
        plugin_dir: str,
        manifest: Dict[str, Any],
        plugin_path: str,
    ) -> Optional[PluginMeta]:
        """加载单个插件"""
        # 确保兼容层导入钩子已安装（旧版插件可能 import src.plugin_system）
        self._ensure_compat_hook()

        # 动态导入插件模块
        module_name = f"_maibot_plugin_{plugin_id}"
        spec = importlib.util.spec_from_file_location(module_name, plugin_path)
        if spec is None or spec.loader is None:
            logger.error(f"无法创建模块 spec: {plugin_path}")
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # 优先使用新版 create_plugin 工厂函数
        create_plugin = getattr(module, "create_plugin", None)
        if create_plugin is not None:
            instance = create_plugin()
            logger.info(f"插件 {plugin_id} v{manifest.get('version', '?')} 加载成功")
            return PluginMeta(
                plugin_id=plugin_id,
                plugin_dir=plugin_dir,
                plugin_instance=instance,
                manifest=manifest,
            )

        # 回退：检测旧版 @register_plugin 标记的 BasePlugin 子类
        instance = self._try_load_legacy_plugin(module, plugin_id)
        if instance is not None:
            logger.info(
                f"插件 {plugin_id} v{manifest.get('version', '?')} 通过旧版兼容层加载成功（请尽快迁移到 maibot_sdk）"
            )
            return PluginMeta(
                plugin_id=plugin_id,
                plugin_dir=plugin_dir,
                plugin_instance=instance,
                manifest=manifest,
            )

        logger.error(f"插件 {plugin_id} 缺少 create_plugin 工厂函数且未检测到旧版 BasePlugin")
        return None

    # ──── 旧版插件兼容 ────────────────────────────────────────

    def _ensure_compat_hook(self) -> None:
        """安装旧版 src.plugin_system 导入钩子（幂等）"""
        if self._compat_hook_installed:
            return
        try:
            from maibot_sdk.compat._import_hook import install_hook

            install_hook()
            self._compat_hook_installed = True
        except ImportError:
            logger.debug("maibot_sdk.compat 不可用，跳过导入钩子安装")

    @staticmethod
    def _try_load_legacy_plugin(module: Any, plugin_id: str) -> Optional[Any]:
        """尝试从模块中发现旧版 BasePlugin 子类并包装为 LegacyPluginAdapter"""
        # 方式 1: @register_plugin 装饰器设置的标记
        legacy_cls = getattr(module, "_legacy_plugin_class", None)

        # 方式 2: 扫描模块中所有 BasePlugin 子类
        if legacy_cls is None:
            try:
                from maibot_sdk.compat.base.base_plugin import BasePlugin as LegacyBasePlugin
            except ImportError:
                return None

            for attr_name in dir(module):
                obj = getattr(module, attr_name, None)
                if isinstance(obj, type) and issubclass(obj, LegacyBasePlugin) and obj is not LegacyBasePlugin:
                    legacy_cls = obj
                    break

        if legacy_cls is None:
            return None

        try:
            from maibot_sdk.compat.legacy_adapter import LegacyPluginAdapter

            legacy_instance = legacy_cls()
            return LegacyPluginAdapter(legacy_instance)
        except Exception as e:
            logger.error(f"旧版插件 {plugin_id} 适配失败: {e}", exc_info=True)
            return None
