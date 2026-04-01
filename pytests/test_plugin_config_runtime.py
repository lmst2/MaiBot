"""插件配置运行时测试。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Tuple, cast

import tomllib

import pytest

from src.plugin_runtime.component_query import component_query_service
from src.plugin_runtime.protocol.envelope import (
    Envelope,
    MessageType,
    RegisterPluginPayload,
    ValidatePluginConfigPayload,
)
from src.plugin_runtime.runner.runner_main import PluginRunner
from src.webui.routers.plugin.config_routes import update_plugin_config
from src.webui.routers.plugin.schemas import UpdatePluginConfigRequest


class _DemoConfigPlugin:
    """用于测试 Runner 配置归一化流程的伪插件。"""

    def __init__(self) -> None:
        """初始化测试插件状态。"""

        self.received_config: Dict[str, Any] = {}

    def normalize_plugin_config(self, config_data: Optional[Mapping[str, Any]]) -> Tuple[Dict[str, Any], bool]:
        """补齐测试插件的默认配置。

        Args:
            config_data: 原始配置数据。

        Returns:
            Tuple[Dict[str, Any], bool]: 补齐后的配置，以及是否发生变更。
        """

        current_config = dict(config_data or {})
        plugin_section = dict(current_config.get("plugin", {}))
        changed = "retry_count" not in plugin_section
        plugin_section.setdefault("enabled", True)
        plugin_section.setdefault("retry_count", 3)
        return {"plugin": plugin_section}, changed

    def set_plugin_config(self, config: Dict[str, Any]) -> None:
        """记录 Runner 注入的配置内容。

        Args:
            config: 当前最新配置。
        """

        self.received_config = config


class _StrictConfigPlugin:
    """用于测试配置校验错误的伪插件。"""

    def normalize_plugin_config(self, config_data: Optional[Mapping[str, Any]]) -> Tuple[Dict[str, Any], bool]:
        """校验重试次数不能为负数。

        Args:
            config_data: 原始配置数据。

        Returns:
            Tuple[Dict[str, Any], bool]: 规范化配置结果。

        Raises:
            ValueError: 当重试次数为负数时抛出。
        """

        current_config = dict(config_data or {})
        plugin_section = dict(current_config.get("plugin", {}))
        retry_count = int(plugin_section.get("retry_count", 0))
        if retry_count < 0:
            raise ValueError("重试次数不能小于 0")
        plugin_section.setdefault("enabled", True)
        return {"plugin": plugin_section}, False

    def set_plugin_config(self, config: Dict[str, Any]) -> None:
        """兼容 Runner 配置注入接口。

        Args:
            config: 当前配置字典。
        """

        del config


def test_runner_apply_plugin_config_generates_config_file(tmp_path: Path) -> None:
    """Runner 注入配置时应自动补齐并落盘 config.toml。"""

    plugin = _DemoConfigPlugin()
    runner = PluginRunner(
        host_address="ipc://unused",
        session_token="session-token",
        plugin_dirs=[],
    )
    meta = SimpleNamespace(plugin_id="demo.plugin", plugin_dir=str(tmp_path), instance=plugin)

    runner._apply_plugin_config(cast(Any, meta), config_data={"plugin": {"enabled": False}})

    config_path = tmp_path / "config.toml"
    assert config_path.exists()
    assert plugin.received_config == {"plugin": {"enabled": False, "retry_count": 3}}

    with config_path.open("rb") as handle:
        saved_config = tomllib.load(handle)
    assert saved_config == {"plugin": {"enabled": False, "retry_count": 3}}


def test_component_query_service_returns_plugin_config_schema(monkeypatch: Any) -> None:
    """组件查询服务应支持按插件 ID 返回配置 Schema。"""

    payload = RegisterPluginPayload(
        plugin_id="demo.plugin",
        plugin_version="1.0.0",
        default_config={"plugin": {"enabled": True}},
        config_schema={
            "plugin_id": "demo.plugin",
            "plugin_info": {
                "name": "Demo",
                "version": "1.0.0",
                "description": "",
                "author": "",
            },
            "sections": {"plugin": {"fields": {}}},
            "layout": {"type": "auto", "tabs": []},
        },
    )
    fake_supervisor = SimpleNamespace(_registered_plugins={"demo.plugin": payload})
    fake_manager = SimpleNamespace(_get_supervisor_for_plugin=lambda plugin_id: fake_supervisor)

    monkeypatch.setattr(
        type(component_query_service),
        "_get_runtime_manager",
        staticmethod(lambda: fake_manager),
    )

    assert component_query_service.get_plugin_config_schema("demo.plugin") == payload.config_schema
    assert component_query_service.get_plugin_default_config("demo.plugin") == payload.default_config


@pytest.mark.asyncio
async def test_runner_validate_plugin_config_handler_returns_normalized_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runner 应返回插件模型归一化后的配置。"""

    plugin = _DemoConfigPlugin()
    runner = PluginRunner(
        host_address="ipc://unused",
        session_token="session-token",
        plugin_dirs=[],
    )
    meta = SimpleNamespace(plugin_id="demo.plugin", plugin_dir="", instance=plugin)
    monkeypatch.setattr(runner._loader, "get_plugin", lambda plugin_id: meta if plugin_id == "demo.plugin" else None)

    envelope = Envelope(
        request_id=1,
        message_type=MessageType.REQUEST,
        method="plugin.validate_config",
        plugin_id="demo.plugin",
        payload=ValidatePluginConfigPayload(config_data={"plugin": {"enabled": False}}).model_dump(),
    )

    response = await runner._handle_validate_plugin_config(envelope)

    assert response.error is None
    assert response.payload["success"] is True
    assert response.payload["normalized_config"] == {"plugin": {"enabled": False, "retry_count": 3}}


@pytest.mark.asyncio
async def test_runner_validate_plugin_config_handler_returns_error_on_invalid_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runner 应在插件拒绝配置时返回错误响应。"""

    plugin = _StrictConfigPlugin()
    runner = PluginRunner(
        host_address="ipc://unused",
        session_token="session-token",
        plugin_dirs=[],
    )
    meta = SimpleNamespace(plugin_id="demo.plugin", plugin_dir="", instance=plugin)
    monkeypatch.setattr(runner._loader, "get_plugin", lambda plugin_id: meta if plugin_id == "demo.plugin" else None)

    envelope = Envelope(
        request_id=1,
        message_type=MessageType.REQUEST,
        method="plugin.validate_config",
        plugin_id="demo.plugin",
        payload=ValidatePluginConfigPayload(config_data={"plugin": {"retry_count": -1}}).model_dump(),
    )

    response = await runner._handle_validate_plugin_config(envelope)

    assert response.error is not None
    assert response.error["message"] == "重试次数不能小于 0"


@pytest.mark.asyncio
async def test_update_plugin_config_prefers_runtime_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """WebUI 保存插件配置时应优先使用运行时校验结果。"""

    config_path = tmp_path / "config.toml"

    async def _mock_validate_plugin_config(plugin_id: str, config_data: Dict[str, Any]) -> Dict[str, Any] | None:
        """返回运行时归一化后的配置。

        Args:
            plugin_id: 插件 ID。
            config_data: 原始配置。

        Returns:
            Dict[str, Any] | None: 归一化后的配置。
        """

        assert plugin_id == "demo.plugin"
        assert config_data == {"plugin": {"enabled": False}}
        return {"plugin": {"enabled": False, "retry_count": 3}}

    fake_runtime_manager = SimpleNamespace(validate_plugin_config=_mock_validate_plugin_config)

    monkeypatch.setattr(
        "src.webui.routers.plugin.config_routes.require_plugin_token",
        lambda session: session or "session-token",
    )
    monkeypatch.setattr(
        "src.webui.routers.plugin.config_routes.find_plugin_path_by_id",
        lambda plugin_id: tmp_path if plugin_id == "demo.plugin" else None,
    )
    monkeypatch.setattr(
        "src.plugin_runtime.integration.get_plugin_runtime_manager",
        lambda: fake_runtime_manager,
    )

    response = await update_plugin_config(
        "demo.plugin",
        UpdatePluginConfigRequest(config={"plugin.enabled": False}),
        maibot_session="session-token",
    )

    assert response["success"] is True
    with config_path.open("rb") as handle:
        saved_config = tomllib.load(handle)
    assert saved_config == {"plugin": {"enabled": False, "retry_count": 3}}
