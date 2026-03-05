"""Supervisor - 插件生命周期管理

负责：
1. 拉起 Runner 子进程
2. 健康检查
3. 熔断与恢复
4. 代码热重载（generation 切换）
5. 优雅关停
"""

from typing import Any

import asyncio
import logging
import os
import sys

from src.plugin_runtime.host.capability_service import CapabilityService
from src.plugin_runtime.host.circuit_breaker import CircuitBreakerRegistry
from src.plugin_runtime.host.policy_engine import PolicyEngine
from src.plugin_runtime.host.rpc_server import RPCServer
from src.plugin_runtime.protocol.envelope import (
    Envelope,
    HealthPayload,
    RegisterComponentsPayload,
    ShutdownPayload,
)
from src.plugin_runtime.protocol.errors import ErrorCode, RPCError
from src.plugin_runtime.transport.factory import create_transport_server

logger = logging.getLogger("plugin_runtime.host.supervisor")


class PluginSupervisor:
    """插件 Supervisor

    Host 端的核心管理器，负责整个插件 Runner 进程的生命周期。
    """

    def __init__(
        self,
        plugin_dirs: list[str] | None = None,
        socket_path: str | None = None,
        health_check_interval_sec: float = 30.0,
        use_json_codec: bool = False,
    ):
        self._plugin_dirs = plugin_dirs or []
        self._health_interval = health_check_interval_sec

        # 基础设施
        self._transport = create_transport_server(socket_path=socket_path)
        self._policy = PolicyEngine()
        self._breakers = CircuitBreakerRegistry()
        self._capability_service = CapabilityService(self._policy)

        # 编解码
        from src.plugin_runtime.protocol.codec import create_codec
        codec = create_codec(use_json=use_json_codec)

        self._rpc_server = RPCServer(
            transport=self._transport,
            codec=codec,
        )

        # Runner 子进程
        self._runner_process: asyncio.subprocess.Process | None = None
        self._runner_generation: int = 0

        # 已注册的插件组件信息
        self._registered_plugins: dict[str, RegisterComponentsPayload] = {}

        # 后台任务
        self._health_task: asyncio.Task | None = None
        self._running = False

        # 注册内部 RPC 方法
        self._register_internal_methods()

    @property
    def policy_engine(self) -> PolicyEngine:
        return self._policy

    @property
    def capability_service(self) -> CapabilityService:
        return self._capability_service

    @property
    def rpc_server(self) -> RPCServer:
        return self._rpc_server

    async def start(self) -> None:
        """启动 Supervisor

        1. 启动 RPC Server
        2. 拉起 Runner 子进程
        3. 启动健康检查
        """
        self._running = True

        # 启动 RPC Server
        await self._rpc_server.start()

        # 拉起 Runner 进程
        await self._spawn_runner()

        # 启动健康检查
        self._health_task = asyncio.create_task(self._health_check_loop())

        logger.info("PluginSupervisor 已启动")

    async def stop(self) -> None:
        """停止 Supervisor"""
        self._running = False

        # 停止健康检查
        if self._health_task:
            self._health_task.cancel()
            self._health_task = None

        # 优雅关停 Runner
        await self._shutdown_runner()

        # 停止 RPC Server
        await self._rpc_server.stop()

        logger.info("PluginSupervisor 已停止")

    async def invoke_plugin(
        self,
        method: str,
        plugin_id: str,
        component_name: str,
        args: dict[str, Any] | None = None,
        timeout_ms: int = 30000,
    ) -> Envelope:
        """调用插件组件

        由主进程业务逻辑调用，通过 RPC 转发给 Runner。
        """
        # 熔断检查
        breaker = self._breakers.get(plugin_id)
        if not breaker.allow_request():
            raise RPCError(ErrorCode.E_PLUGIN_CRASHED, f"插件 {plugin_id} 已被熔断")

        try:
            response = await self._rpc_server.send_request(
                method=method,
                plugin_id=plugin_id,
                payload={
                    "component_name": component_name,
                    "args": args or {},
                },
                timeout_ms=timeout_ms,
            )
            breaker.record_success()
            return response
        except RPCError:
            breaker.record_failure()
            raise

    async def reload_plugins(self, reason: str = "manual") -> None:
        """热重载所有插件（进程级 generation 切换）

        1. 拉起新 Runner
        2. 等待新 Runner 完成注册和健康检查
        3. 关停旧 Runner
        """
        logger.info(f"开始热重载插件，原因: {reason}")

        # 保存旧进程引用
        old_process = self._runner_process

        # 拉起新 Runner
        await self._spawn_runner()

        # 等待新 Runner 连接并完成握手
        for _ in range(30):  # 最多等待 30 秒
            if self._rpc_server.is_connected:
                break
            await asyncio.sleep(1.0)
        else:
            logger.error("新 Runner 连接超时，回滚")
            # 回滚：终止新进程
            if self._runner_process and self._runner_process != old_process:
                self._runner_process.terminate()
            self._runner_process = old_process
            return

        # 健康检查
        try:
            resp = await self._rpc_server.send_request("plugin.health", timeout_ms=5000)
            health = HealthPayload.model_validate(resp.payload)
            if not health.healthy:
                raise RPCError(ErrorCode.E_PLUGIN_CRASHED, "新 Runner 健康检查失败")
        except Exception as e:
            logger.error(f"新 Runner 健康检查失败: {e}，回滚")
            if self._runner_process and self._runner_process != old_process:
                self._runner_process.terminate()
            self._runner_process = old_process
            return

        # 关停旧 Runner
        if old_process and old_process.returncode is None:
            try:
                old_process.terminate()
                await asyncio.wait_for(old_process.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                old_process.kill()

        logger.info("热重载完成")

    # ─── 内部方法 ──────────────────────────────────────────────

    def _register_internal_methods(self) -> None:
        """注册 Host 端的 RPC 方法处理器"""
        # Runner -> Host 的能力调用统一走 capability_service
        self._rpc_server.register_method("cap.request", self._capability_service.handle_capability_request)
        # 插件注册
        self._rpc_server.register_method("plugin.register_components", self._handle_register_components)

    async def _handle_register_components(self, envelope: Envelope) -> Envelope:
        """处理插件组件注册请求"""
        try:
            reg = RegisterComponentsPayload.model_validate(envelope.payload)
        except Exception as e:
            return envelope.make_error_response(ErrorCode.E_BAD_PAYLOAD.value, str(e))

        # 记录注册信息
        self._registered_plugins[reg.plugin_id] = reg

        # 在策略引擎中注册插件
        self._policy.register_plugin(
            plugin_id=reg.plugin_id,
            generation=envelope.generation,
            capabilities=reg.capabilities_required,
        )

        logger.info(
            f"插件 {reg.plugin_id} v{reg.plugin_version} 注册成功，"
            f"组件数: {len(reg.components)}, 能力需求: {reg.capabilities_required}"
        )

        return envelope.make_response(payload={"accepted": True})

    async def _spawn_runner(self) -> None:
        """拉起 Runner 子进程"""
        runner_module = "src.plugin_runtime.runner.runner_main"
        address = self._transport.get_address()
        token = self._rpc_server.session_token

        env = os.environ.copy()
        env["MAIBOT_IPC_ADDRESS"] = address
        env["MAIBOT_SESSION_TOKEN"] = token
        env["MAIBOT_PLUGIN_DIRS"] = os.pathsep.join(self._plugin_dirs)

        self._runner_process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", runner_module,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._runner_generation += 1
        logger.info(f"Runner 子进程已启动: pid={self._runner_process.pid}, generation={self._runner_generation}")

    async def _shutdown_runner(self) -> None:
        """优雅关停 Runner"""
        if not self._runner_process or self._runner_process.returncode is not None:
            return

        # 发送 prepare_shutdown
        try:
            if self._rpc_server.is_connected:
                shutdown_payload = ShutdownPayload(reason="host_shutdown", drain_timeout_ms=5000)
                await self._rpc_server.send_request(
                    "plugin.prepare_shutdown",
                    payload=shutdown_payload.model_dump(),
                    timeout_ms=5000,
                )
                await self._rpc_server.send_request(
                    "plugin.shutdown",
                    payload=shutdown_payload.model_dump(),
                    timeout_ms=5000,
                )
        except Exception as e:
            logger.warning(f"发送关停命令失败: {e}")

        # 等待进程退出
        try:
            await asyncio.wait_for(self._runner_process.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("Runner 未在超时内退出，强制终止")
            self._runner_process.kill()
            await self._runner_process.wait()

    async def _health_check_loop(self) -> None:
        """周期性健康检查"""
        while self._running:
            await asyncio.sleep(self._health_interval)

            if not self._rpc_server.is_connected:
                logger.warning("Runner 未连接，跳过健康检查")
                continue

            try:
                resp = await self._rpc_server.send_request("plugin.health", timeout_ms=5000)
                health = HealthPayload.model_validate(resp.payload)
                if not health.healthy:
                    logger.warning(f"Runner 健康检查异常: {health}")
            except RPCError as e:
                logger.error(f"健康检查失败: {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"健康检查异常: {e}")
