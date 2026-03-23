from types import SimpleNamespace
from typing import Any

import pytest

from src.core.component_registry import component_registry as core_component_registry
from src.plugin_runtime.host.supervisor import PluginSupervisor
from src.plugin_runtime.protocol.envelope import ComponentDeclaration, RegisterPluginPayload


def _build_action_payload(plugin_id: str, action_name: str) -> RegisterPluginPayload:
    """构造用于测试的 runtime Action 注册载荷。

    Args:
        plugin_id: 插件 ID。
        action_name: Action 名称。

    Returns:
        RegisterPluginPayload: 测试用注册载荷。
    """
    return RegisterPluginPayload(
        plugin_id=plugin_id,
        plugin_version="1.0.0",
        components=[
            ComponentDeclaration(
                name=action_name,
                component_type="ACTION",
                plugin_id=plugin_id,
                metadata={
                    "description": "发送一个测试回复",
                    "enabled": True,
                    "activation_type": "keyword",
                    "activation_probability": 0.25,
                    "activation_keywords": ["测试", "hello"],
                    "action_parameters": {"target": "目标对象"},
                    "action_require": ["需要发送回复时使用"],
                    "associated_types": ["text"],
                    "parallel_action": True,
                },
            )
        ],
    )


@pytest.mark.asyncio
async def test_runtime_actions_are_mirrored_into_core_registry_and_invoked(monkeypatch: pytest.MonkeyPatch) -> None:
    """运行时 Action 应镜像到旧核心注册表，并可由旧 Planner 执行。"""
    plugin_id = "runtime_action_bridge_plugin"
    action_name = "runtime_action_bridge_test"
    payload = _build_action_payload(plugin_id=plugin_id, action_name=action_name)
    supervisor = PluginSupervisor(plugin_dirs=[])
    captured: dict[str, Any] = {}

    core_component_registry.remove_action(action_name)

    async def fake_invoke_plugin(
        method: str,
        plugin_id: str,
        component_name: str,
        args: dict[str, Any] | None = None,
        timeout_ms: int = 30000,
    ) -> Any:
        """模拟 plugin runtime Action 调用。

        Args:
            method: RPC 方法名。
            plugin_id: 插件 ID。
            component_name: 组件名称。
            args: 调用参数。
            timeout_ms: RPC 超时时间。

        Returns:
            Any: 伪造的 RPC 响应对象。
        """
        captured["method"] = method
        captured["plugin_id"] = plugin_id
        captured["component_name"] = component_name
        captured["args"] = args or {}
        captured["timeout_ms"] = timeout_ms
        return SimpleNamespace(payload={"success": True, "result": (True, "runtime action executed")})

    monkeypatch.setattr(supervisor, "invoke_plugin", fake_invoke_plugin)

    try:
        supervisor._mirror_runtime_actions_to_core_registry(payload)

        action_info = core_component_registry.get_action_info(action_name)
        assert action_info is not None
        assert action_info.plugin_name == plugin_id
        assert action_info.description == "发送一个测试回复"
        assert action_info.activation_keywords == ["测试", "hello"]
        assert action_info.random_activation_probability == 0.25
        assert action_info.parallel_action is True

        executor = core_component_registry.get_action_executor(action_name)
        assert executor is not None

        success, reason = await executor(
            action_data={"target": "MaiBot"},
            action_reasoning="当前适合使用这个动作",
            cycle_timers={"planner": 0.1},
            thinking_id="tid-1",
            chat_stream=SimpleNamespace(session_id="stream-1"),
            log_prefix="[test]",
            shutting_down=False,
            plugin_config={"enabled": True},
        )

        assert success is True
        assert reason == "runtime action executed"
        assert captured["method"] == "plugin.invoke_action"
        assert captured["plugin_id"] == plugin_id
        assert captured["component_name"] == action_name
        assert captured["args"]["stream_id"] == "stream-1"
        assert captured["args"]["chat_id"] == "stream-1"
        assert captured["args"]["reasoning"] == "当前适合使用这个动作"
        assert captured["args"]["target"] == "MaiBot"
        assert captured["args"]["action_data"] == {"target": "MaiBot"}
    finally:
        supervisor._remove_core_action_mirrors(plugin_id)
        core_component_registry.remove_action(action_name)


def test_clear_runner_state_removes_mirrored_runtime_actions() -> None:
    """清理 Runner 状态时应同步移除旧核心注册表中的镜像 Action。"""
    plugin_id = "runtime_action_bridge_cleanup_plugin"
    action_name = "runtime_action_bridge_cleanup_test"
    payload = _build_action_payload(plugin_id=plugin_id, action_name=action_name)
    supervisor = PluginSupervisor(plugin_dirs=[])

    core_component_registry.remove_action(action_name)

    supervisor._mirror_runtime_actions_to_core_registry(payload)
    assert core_component_registry.get_action_info(action_name) is not None

    supervisor._clear_runner_state()

    assert core_component_registry.get_action_info(action_name) is None
