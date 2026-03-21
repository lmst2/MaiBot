"""适配器运行时状态同步测试。"""

from typing import Any, Dict

import pytest

from src.platform_io.manager import PlatformIOManager
from src.platform_io.types import RouteKey
from src.plugin_runtime.host.supervisor import PluginSupervisor
from src.plugin_runtime.protocol.envelope import (
    AdapterDeclarationPayload,
    Envelope,
    MessageType,
)


def _make_request(plugin_id: str, payload: Dict[str, Any]) -> Envelope:
    """构造一个适配器状态更新 RPC 请求。

    Args:
        plugin_id: 目标适配器插件 ID。
        payload: 请求载荷。

    Returns:
        Envelope: 标准 RPC 请求信封。
    """
    return Envelope(
        request_id=1,
        message_type=MessageType.REQUEST,
        method="host.update_adapter_state",
        plugin_id=plugin_id,
        payload=payload,
    )


@pytest.mark.asyncio
async def test_adapter_runtime_state_binds_and_unbinds_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    """连接建立后应绑定路由，断开后应撤销路由。"""
    import src.plugin_runtime.host.supervisor as supervisor_module

    platform_io_manager = PlatformIOManager()
    monkeypatch.setattr(supervisor_module, "get_platform_io_manager", lambda: platform_io_manager)

    supervisor = PluginSupervisor(plugin_dirs=[])
    adapter = AdapterDeclarationPayload(platform="qq", protocol="napcat")
    await supervisor._register_adapter_driver("napcat_adapter_builtin", adapter)

    response = await supervisor._handle_update_adapter_state(
        _make_request(
            "napcat_adapter_builtin",
            {
                "connected": True,
                "account_id": "10001",
                "scope": "",
                "metadata": {},
            },
        )
    )

    assert response.error is None
    assert response.payload["accepted"] is True
    assert (
        platform_io_manager.route_table.get_active_binding(
            RouteKey(platform="qq", account_id="10001"),
            exact_only=True,
        ).driver_id
        == "adapter:napcat_adapter_builtin"
    )
    assert (
        platform_io_manager.route_table.get_active_binding(
            RouteKey(platform="qq"),
            exact_only=True,
        ).driver_id
        == "adapter:napcat_adapter_builtin"
    )

    response = await supervisor._handle_update_adapter_state(
        _make_request(
            "napcat_adapter_builtin",
            {
                "connected": False,
                "account_id": "",
                "scope": "",
                "metadata": {},
            },
        )
    )

    assert response.error is None
    assert response.payload["accepted"] is True
    assert platform_io_manager.route_table.get_active_binding(
        RouteKey(platform="qq", account_id="10001"),
        exact_only=True,
    ) is None
    assert platform_io_manager.route_table.get_active_binding(RouteKey(platform="qq"), exact_only=True) is None


@pytest.mark.asyncio
async def test_platform_default_route_is_removed_when_multiple_exact_routes_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一平台存在多个精确路由时不应保留默认平台路由。"""
    import src.plugin_runtime.host.supervisor as supervisor_module

    platform_io_manager = PlatformIOManager()
    monkeypatch.setattr(supervisor_module, "get_platform_io_manager", lambda: platform_io_manager)

    supervisor = PluginSupervisor(plugin_dirs=[])
    adapter = AdapterDeclarationPayload(platform="qq", protocol="napcat")
    await supervisor._register_adapter_driver("adapter_a", adapter)
    await supervisor._register_adapter_driver("adapter_b", adapter)

    await supervisor._handle_update_adapter_state(
        _make_request(
            "adapter_a",
            {
                "connected": True,
                "account_id": "10001",
                "scope": "",
                "metadata": {},
            },
        )
    )
    assert (
        platform_io_manager.route_table.get_active_binding(
            RouteKey(platform="qq"),
            exact_only=True,
        ).driver_id
        == "adapter:adapter_a"
    )

    await supervisor._handle_update_adapter_state(
        _make_request(
            "adapter_b",
            {
                "connected": True,
                "account_id": "10002",
                "scope": "",
                "metadata": {},
            },
        )
    )
    assert platform_io_manager.route_table.get_active_binding(RouteKey(platform="qq"), exact_only=True) is None

    await supervisor._handle_update_adapter_state(
        _make_request(
            "adapter_b",
            {
                "connected": False,
                "account_id": "",
                "scope": "",
                "metadata": {},
            },
        )
    )
    assert (
        platform_io_manager.route_table.get_active_binding(
            RouteKey(platform="qq"),
            exact_only=True,
        ).driver_id
        == "adapter:adapter_a"
    )
