"""NapCat 插件与新 SDK 对接测试。"""

from importlib import import_module, util
from pathlib import Path
from typing import Any, Dict, List, Tuple

import logging
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLUGINS_ROOT = PROJECT_ROOT / "plugins"
SDK_ROOT = PROJECT_ROOT / "packages" / "maibot-plugin-sdk"
NAPCAT_PLUGIN_DIR = PLUGINS_ROOT / "MaiBot-Napcat-Adapter"
NAPCAT_TEST_MODULE = "_test_napcat_adapter"

for import_path in (str(SDK_ROOT),):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)


class _FakeGatewayCapability:
    """用于捕获消息网关状态上报的测试替身。"""

    def __init__(self) -> None:
        """初始化测试替身。"""

        self.calls: List[Dict[str, Any]] = []

    async def update_state(
        self,
        gateway_name: str,
        *,
        ready: bool,
        platform: str = "",
        account_id: str = "",
        scope: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> bool:
        """记录一次状态上报请求。

        Args:
            gateway_name: 网关组件名称。
            ready: 当前是否就绪。
            platform: 平台名称。
            account_id: 账号 ID。
            scope: 路由作用域。
            metadata: 附加元数据。

        Returns:
            bool: 始终返回 ``True``，模拟 Host 接受状态更新。
        """

        self.calls.append(
            {
                "gateway_name": gateway_name,
                "ready": ready,
                "platform": platform,
                "account_id": account_id,
                "scope": scope,
                "metadata": metadata or {},
            }
        )
        return True


def _load_napcat_sdk_modules() -> Tuple[Any, Any, Any, Any]:
    """动态加载 NapCat 插件测试所需的模块。

    Returns:
        tuple[Any, Any, Any, Any]:
            依次返回常量模块、配置模块、插件模块和运行时状态模块。
    """

    if NAPCAT_TEST_MODULE not in sys.modules:
        plugin_path = NAPCAT_PLUGIN_DIR / "plugin.py"
        spec = util.spec_from_file_location(
            NAPCAT_TEST_MODULE,
            plugin_path,
            submodule_search_locations=[str(NAPCAT_PLUGIN_DIR)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"无法为 NapCat 插件创建模块规格: {plugin_path}")

        module = util.module_from_spec(spec)
        sys.modules[NAPCAT_TEST_MODULE] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(NAPCAT_TEST_MODULE, None)
            raise

    return (
        import_module(f"{NAPCAT_TEST_MODULE}.constants"),
        import_module(f"{NAPCAT_TEST_MODULE}.config"),
        import_module(f"{NAPCAT_TEST_MODULE}.plugin"),
        import_module(f"{NAPCAT_TEST_MODULE}.runtime_state"),
    )


def _load_napcat_sdk_symbols() -> Tuple[Any, Any, Any, Any]:
    """动态加载 NapCat 插件测试所需的符号。

    Returns:
        tuple[Any, Any, Any, Any]:
            依次返回网关名常量、配置类、插件类和运行时状态管理器类。
    """

    constants_module, config_module, plugin_module, runtime_state_module = _load_napcat_sdk_modules()
    return (
        constants_module.NAPCAT_GATEWAY_NAME,
        config_module.NapCatServerConfig,
        plugin_module.NapCatAdapterPlugin,
        runtime_state_module.NapCatRuntimeStateManager,
    )


def test_napcat_plugin_collects_duplex_message_gateway() -> None:
    """NapCat 插件应声明新的双工消息网关组件。"""

    napcat_gateway_name, _napcat_server_config, napcat_plugin_cls, _runtime_state_cls = _load_napcat_sdk_symbols()
    plugin = napcat_plugin_cls()
    components = plugin.get_components()
    gateway_components = [
        component
        for component in components
        if component.get("type") == "MESSAGE_GATEWAY"
    ]

    assert len(gateway_components) == 1
    gateway_component = gateway_components[0]
    assert gateway_component["name"] == napcat_gateway_name
    assert gateway_component["metadata"]["route_type"] == "duplex"
    assert gateway_component["metadata"]["platform"] == "qq"
    assert gateway_component["metadata"]["protocol"] == "napcat"


def test_napcat_plugin_uses_sdk_config_model() -> None:
    """NapCat 插件应声明 SDK 配置模型并暴露默认配置与 Schema。"""

    constants_module, _config_module, plugin_module, _runtime_state_module = _load_napcat_sdk_modules()
    plugin = plugin_module.NapCatAdapterPlugin()

    default_config = plugin.get_default_config()
    schema = plugin.get_webui_config_schema(plugin_id="maibot-team.napcat-adapter")

    assert default_config["plugin"]["config_version"] == constants_module.SUPPORTED_CONFIG_VERSION
    assert default_config["chat"]["ban_qq_bot"] is False
    assert default_config["filters"]["ignore_self_message"] is True
    assert schema["plugin_id"] == "maibot-team.napcat-adapter"
    assert schema["sections"]["chat"]["fields"]["group_list"]["type"] == "array"
    assert schema["sections"]["chat"]["fields"]["group_list_type"]["choices"] == ["whitelist", "blacklist"]


def test_napcat_plugin_normalizes_legacy_config_values() -> None:
    """NapCat 插件应兼容旧配置字段并输出规范化结果。"""

    constants_module, _config_module, plugin_module, _runtime_state_module = _load_napcat_sdk_modules()
    plugin = plugin_module.NapCatAdapterPlugin()

    plugin.set_plugin_config(
        {
            "plugin": {"enabled": True, "config_version": ""},
            "connection": {
                "access_token": "secret-token",
                "heartbeat_sec": "45",
                "ws_url": "ws://10.0.0.8:3012/onebot/v11/ws",
            },
            "chat": {
                "ban_qq_bot": True,
                "ban_user_id": ["42", 42, ""],
                "group_list": [123, " 456 ", None, "123"],
                "group_list_type": "whitelist",
                "private_list": "invalid",
                "private_list_type": "unexpected",
            },
            "filters": {"ignore_self_message": True},
        }
    )

    config_data = plugin.get_plugin_config_data()

    assert "connection" not in config_data
    assert config_data["plugin"]["config_version"] == constants_module.SUPPORTED_CONFIG_VERSION
    assert config_data["napcat_server"]["host"] == "10.0.0.8"
    assert config_data["napcat_server"]["port"] == 3012
    assert config_data["napcat_server"]["token"] == "secret-token"
    assert config_data["napcat_server"]["heartbeat_interval"] == 45.0
    assert config_data["chat"]["group_list"] == ["123", "456"]
    assert config_data["chat"]["private_list"] == []
    assert config_data["chat"]["private_list_type"] == constants_module.DEFAULT_CHAT_LIST_TYPE
    assert plugin.config.napcat_server.build_ws_url() == "ws://10.0.0.8:3012"


@pytest.mark.asyncio
async def test_runtime_state_reports_via_gateway_capability() -> None:
    """NapCat 运行时状态应通过新的消息网关能力上报。"""

    napcat_gateway_name, napcat_server_config_cls, _napcat_plugin_cls, runtime_state_cls = _load_napcat_sdk_symbols()
    gateway_capability = _FakeGatewayCapability()
    runtime_state_manager = runtime_state_cls(
        gateway_capability=gateway_capability,
        logger=logging.getLogger("test.napcat_adapter"),
        gateway_name=napcat_gateway_name,
    )

    connected = await runtime_state_manager.report_connected(
        "10001",
        napcat_server_config_cls(connection_id="primary"),
    )
    await runtime_state_manager.report_disconnected()

    assert connected is True
    assert gateway_capability.calls[0]["gateway_name"] == napcat_gateway_name
    assert gateway_capability.calls[0]["ready"] is True
    assert gateway_capability.calls[0]["platform"] == "qq"
    assert gateway_capability.calls[0]["account_id"] == "10001"
    assert gateway_capability.calls[0]["scope"] == "primary"
    assert gateway_capability.calls[1]["gateway_name"] == napcat_gateway_name
    assert gateway_capability.calls[1]["ready"] is False
    assert gateway_capability.calls[1]["platform"] == "qq"
