"""插件加载器

在 Runner 进程中负责发现和加载插件。
插件通过 SDK 编写，不再 import src.*。
"""

from typing import Any

import importlib
import importlib.util
import json
import logging
import os
import sys

logger = logging.getLogger("plugin_runtime.runner.plugin_loader")


class PluginMeta:
    """加载后的插件元数据"""

    def __init__(
        self,
        plugin_id: str,
        plugin_dir: str,
        plugin_instance: Any,
        manifest: dict[str, Any],
    ):
        self.plugin_id = plugin_id
        self.plugin_dir = plugin_dir
        self.instance = plugin_instance
        self.manifest = manifest
        self.version = manifest.get("version", "1.0.0")
        self.capabilities_required = manifest.get("capabilities", [])


class PluginLoader:
    """插件加载器

    扫描插件目录，加载符合 SDK 规范的插件。
    每个插件目录须包含：
    - _manifest.json: 插件元数据
    - plugin.py: 插件入口模块（导出 create_plugin 工厂函数）
    """

    def __init__(self):
        self._loaded_plugins: dict[str, PluginMeta] = {}

    def discover_and_load(self, plugin_dirs: list[str]) -> list[PluginMeta]:
        """扫描多个目录并加载所有插件

        Args:
            plugin_dirs: 插件目录列表

        Returns:
            成功加载的插件元数据列表
        """
        results = []
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
                    meta = self._load_single_plugin(plugin_dir, manifest_path, plugin_path)
                    if meta:
                        self._loaded_plugins[meta.plugin_id] = meta
                        results.append(meta)
                except Exception as e:
                    logger.error(f"加载插件失败 [{plugin_dir}]: {e}", exc_info=True)

        return results

    def get_plugin(self, plugin_id: str) -> PluginMeta | None:
        """获取已加载的插件"""
        return self._loaded_plugins.get(plugin_id)

    def list_plugins(self) -> list[str]:
        """列出所有已加载的插件 ID"""
        return list(self._loaded_plugins.keys())

    def _load_single_plugin(self, plugin_dir: str, manifest_path: str, plugin_path: str) -> PluginMeta | None:
        """加载单个插件"""
        # 1. 读取 manifest
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        plugin_id = os.path.basename(plugin_dir)

        # 2. 动态导入插件模块
        module_name = f"_maibot_plugin_{plugin_id}"
        spec = importlib.util.spec_from_file_location(module_name, plugin_path)
        if spec is None or spec.loader is None:
            logger.error(f"无法创建模块 spec: {plugin_path}")
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # 3. 调用工厂函数创建插件实例
        create_plugin = getattr(module, "create_plugin", None)
        if create_plugin is None:
            logger.error(f"插件 {plugin_id} 缺少 create_plugin 工厂函数")
            return None

        instance = create_plugin()

        logger.info(f"插件 {plugin_id} v{manifest.get('version', '?')} 加载成功")

        return PluginMeta(
            plugin_id=plugin_id,
            plugin_dir=plugin_dir,
            plugin_instance=instance,
            manifest=manifest,
        )
