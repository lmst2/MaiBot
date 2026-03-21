"""NapCat 插件入口行为测试。"""

from pathlib import Path
from typing import List
from types import SimpleNamespace

import importlib
import sys

import pytest


BUILT_IN_PLUGIN_ROOT = Path(__file__).resolve().parents[1] / "src" / "plugins" / "built_in"
if str(BUILT_IN_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(BUILT_IN_PLUGIN_ROOT))

NapCatAdapterPlugin = importlib.import_module("napcat_adapter.plugin").NapCatAdapterPlugin


class DummyLogger:
    """用于测试的轻量日志对象。"""

    def __init__(self) -> None:
        """初始化测试日志对象。"""
        self.debug_messages: List[str] = []

    def debug(self, message: str) -> None:
        """记录调试日志。

        Args:
            message: 待记录的日志内容。
        """
        self.debug_messages.append(message)


@pytest.mark.asyncio
async def test_on_config_update_refreshes_settings_and_restarts(monkeypatch: pytest.MonkeyPatch) -> None:
    """配置更新时应刷新插件配置、清空旧 settings，并触发连接重启。"""
    plugin = NapCatAdapterPlugin()
    plugin._ctx = SimpleNamespace(logger=DummyLogger())
    plugin._settings = object()

    restart_calls: List[dict] = []

    async def fake_restart() -> None:
        """记录一次重启调用。"""
        restart_calls.append(dict(plugin._plugin_config))

    monkeypatch.setattr(plugin, "_restart_connection_if_needed", fake_restart)

    new_config = {
        "plugin": {"enabled": True, "config_version": "0.1.0"},
        "napcat_server": {"host": "127.0.0.1", "port": 3001},
    }
    await plugin.on_config_update(new_config, "v2")

    assert plugin._plugin_config == new_config
    assert plugin._settings is None
    assert restart_calls == [new_config]
    assert plugin.ctx.logger.debug_messages == ["NapCat 适配器收到配置更新通知: v2"]
