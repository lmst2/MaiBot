"""新版本插件运行时与主程序的集成层

提供 PluginRuntimeManager 单例，负责：
1. 管理 PluginSupervisor 的生命周期（启动 / 停止）
2. 将旧系统的 EventType 桥接到新运行时的 event dispatch
3. 将新运行时注册的 command 合并到旧系统的命令查找流程
4. 提供统一的能力实现注册接口，使新插件可以调用主程序功能

在过渡期内，新旧插件系统共存：
- 旧插件继续通过 plugin_manager / component_registry 加载和执行
- 新插件通过 PluginSupervisor + Runner 子进程加载和执行
- 事件和命令在两套系统间桥接
"""

from __future__ import annotations

from typing import Any

import os

from src.common.logger import get_logger

logger = get_logger("plugin_runtime.integration")

# 旧系统 EventType -> 新系统 event_type 字符串映射
_EVENT_TYPE_MAP: dict[str, str] = {
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


class PluginRuntimeManager:
    """新版本插件运行时管理器（单例）

    作为主程序与 PluginSupervisor 之间的桥梁。
    """

    def __init__(self) -> None:
        from src.plugin_runtime.host.supervisor import PluginSupervisor

        self._supervisor: PluginSupervisor | None = None
        self._started: bool = False

    def _get_plugin_dirs(self) -> list[str]:
        """获取新版本插件目录列表

        新版本插件放在 plugins/ 目录中，与旧版本共存。
        只有包含 _manifest.json 的插件目录会被新 Runner 加载。
        """
        dirs: list[str] = []
        for candidate in ("plugins",):
            abs_path: str = os.path.abspath(candidate)
            if os.path.isdir(abs_path):
                dirs.append(abs_path)
        return dirs

    async def start(self) -> None:
        """启动新版本插件运行时

        应在 plugin_manager.load_all_plugins() 之后调用。
        """
        if self._started:
            logger.warning("PluginRuntimeManager 已在运行中，跳过重复启动")
            return

        from src.plugin_runtime.host.supervisor import PluginSupervisor

        plugin_dirs: list[str] = self._get_plugin_dirs()
        if not plugin_dirs:
            logger.info("未找到插件目录，跳过新版本插件运行时启动")
            return

        self._supervisor = PluginSupervisor(plugin_dirs=plugin_dirs)

        # 注册主程序提供的能力实现
        self._register_capability_impls()

        try:
            await self._supervisor.start()
            self._started = True
            logger.info(f"新版本插件运行时已启动，监控目录: {plugin_dirs}")
        except Exception as e:
            logger.error(f"新版本插件运行时启动失败: {e}", exc_info=True)
            self._supervisor = None

    async def stop(self) -> None:
        """停止新版本插件运行时"""
        if not self._started or self._supervisor is None:
            return

        try:
            await self._supervisor.stop()
            logger.info("新版本插件运行时已停止")
        except Exception as e:
            logger.error(f"新版本插件运行时停止失败: {e}", exc_info=True)
        finally:
            self._started = False
            self._supervisor = None

    @property
    def is_running(self) -> bool:
        return self._started

    @property
    def supervisor(self) -> Any:
        """获取底层 Supervisor（供高级用途）"""
        return self._supervisor

    # ─── 事件桥接 ──────────────────────────────────────────────

    async def bridge_event(
        self,
        event_type_value: str,
        message_dict: dict[str, Any] | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> tuple[bool, dict[str, Any] | None]:
        """将旧系统事件转发到新版本插件运行时

        Args:
            event_type_value: 旧 EventType 的 .value（如 "on_message"）
            message_dict: 序列化后的消息字典（MaiMessages 转 dict）
            extra_args: 额外参数

        Returns:
            (continue_flag, modified_message_dict)
        """
        if not self._started or self._supervisor is None:
            return True, None

        new_event_type: str = _EVENT_TYPE_MAP.get(event_type_value, event_type_value)

        try:
            return await self._supervisor.dispatch_event(
                event_type=new_event_type,
                message=message_dict,
                extra_args=extra_args,
            )
        except Exception as e:
            logger.error(f"桥接事件 {new_event_type} 到新运行时失败: {e}", exc_info=True)
            return True, None

    # ─── 命令桥接 ──────────────────────────────────────────────

    def find_command_by_text(self, text: str) -> dict[str, Any] | None:
        """在新版本插件运行时的 ComponentRegistry 中查找命令

        Returns:
            匹配结果字典 {"component": RegisteredComponent, "match": re.Match}
            或 None
        """
        if not self._started or self._supervisor is None:
            return None

        return self._supervisor.component_registry.find_command_by_text(text)

    # ─── 能力实现注册 ──────────────────────────────────────────

    def _register_capability_impls(self) -> None:
        """注册主程序提供的能力实现

        新版本插件通过 cap.request RPC 请求能力调用，
        Host 端的 CapabilityService 需要真正的能力实现来处理这些请求。
        这里注册主程序中可用的功能接口。
        """
        if self._supervisor is None:
            return

        cap_service = self._supervisor.capability_service

        # 注册 send.* 能力
        cap_service.register_capability("send.text", self._cap_send_text)
        cap_service.register_capability("send.emoji", self._cap_send_emoji)
        cap_service.register_capability("send.image", self._cap_send_image)

        # 注册 llm.* 能力
        cap_service.register_capability("llm.generate", self._cap_llm_generate)

        # 注册 config.* 能力
        cap_service.register_capability("config.get", self._cap_config_get)

        logger.debug("已注册主程序能力实现")

    # ─── 能力实现 ──────────────────────────────────────────────

    @staticmethod
    async def _cap_send_text(plugin_id: str, capability: str, args: dict[str, Any]) -> Any:
        """发送文本消息能力实现

        注意: chat_stream 模块已被移除，send.text 能力暂不可用，
        待新的消息发送接口稳定后再接入。
        """
        return {"success": False, "error": "send.text 尚未接入（chat_stream 已移除）"}

    @staticmethod
    async def _cap_send_emoji(plugin_id: str, capability: str, args: dict[str, Any]) -> Any:
        """发送表情能力实现"""
        return {"success": False, "error": "send.emoji 尚未实现"}

    @staticmethod
    async def _cap_send_image(plugin_id: str, capability: str, args: dict[str, Any]) -> Any:
        """发送图片能力实现"""
        return {"success": False, "error": "send.image 尚未实现"}

    @staticmethod
    async def _cap_llm_generate(plugin_id: str, capability: str, args: dict[str, Any]) -> Any:
        """LLM 生成能力实现"""
        return {"success": False, "error": "llm.generate 尚未完全接入，请使用旧系统的 LLM API"}

    @staticmethod
    async def _cap_config_get(plugin_id: str, capability: str, args: dict[str, Any]) -> Any:
        """配置读取能力实现"""
        from src.plugin_system.core import component_registry as old_registry

        plugin_name: str = args.get("plugin_name", plugin_id)
        key: str = args.get("key", "")

        try:
            config = old_registry.get_plugin_config(plugin_name)
            if config is None:
                return {"success": False, "value": None, "error": f"未找到插件 {plugin_name} 的配置"}

            if key:
                parts = key.split(".")
                value: Any = config
                for part in parts:
                    if isinstance(value, dict):
                        value = value.get(part)
                    else:
                        return {"success": False, "value": None, "error": f"配置路径无效: {key}"}
                return {"success": True, "value": value}

            return {"success": True, "value": config}
        except Exception as e:
            return {"success": False, "value": None, "error": str(e)}


# ─── 单例 ──────────────────────────────────────────────────

_manager: PluginRuntimeManager | None = None


def get_plugin_runtime_manager() -> PluginRuntimeManager:
    """获取 PluginRuntimeManager 全局单例"""
    global _manager
    if _manager is None:
        _manager = PluginRuntimeManager()
    return _manager
