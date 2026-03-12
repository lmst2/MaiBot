"""Supervisor - 插件生命周期管理

负责：
1. 拉起 Runner 子进程
2. 健康检查 + 崩溃自动重启
3. 代码热重载（generation 切换）
4. 优雅关停
"""

import logging as stdlib_logging
from typing import Any, Dict, List, Optional, Tuple

import asyncio
import contextlib
import os
import sys

from src.common.logger import get_logger
from src.plugin_runtime.host.capability_service import CapabilityService
from src.plugin_runtime.host.component_registry import ComponentRegistry
from src.plugin_runtime.host.event_dispatcher import EventDispatcher
from src.plugin_runtime.host.policy_engine import PolicyEngine
from src.plugin_runtime.host.rpc_server import RPCServer
from src.plugin_runtime.host.workflow_executor import WorkflowExecutor, WorkflowContext, WorkflowResult
from src.plugin_runtime.protocol.envelope import (
    Envelope,
    HealthPayload,
    LogBatchPayload,
    RegisterComponentsPayload,
    ShutdownPayload,
)
from src.plugin_runtime.protocol.errors import ErrorCode, RPCError
from src.plugin_runtime.transport.factory import create_transport_server

logger = get_logger("plugin_runtime.host.supervisor")


# ─── 日志桥 ──────────────────────────────────────────────────────

class RunnerLogBridge:
    """将 Runner 进程上报的批量日志重放到主进程的 Logger 中。

    Runner 通过 ``runner.log_batch`` IPC 事件批量到达。
    每条 LogEntry 被重建为一个真实的 :class:`logging.LogRecord` 并直接
    调用 ``logging.getLogger(entry.logger_name).handle(record)``，
    从而接入主进程已配置好的 structlog Handler 链。
    """

    async def handle_log_batch(self, envelope: Envelope) -> Envelope:
        """IPC 事件处理器：解析批量日志并重放到主进程 Logger。

        Args:
            envelope: 方法名为 ``runner.log_batch`` 的 IPC 事件信封。

        Returns:
            空响应信封（事件模式下将被忽略）。
        """
        try:
            batch = LogBatchPayload.model_validate(envelope.payload)
        except Exception as exc:
            return envelope.make_error_response(ErrorCode.E_BAD_PAYLOAD.value, str(exc))

        for entry in batch.entries:
            # 重建一个与原始日志尽量相符的 LogRecord
            record = stdlib_logging.LogRecord(
                name=entry.logger_name,
                level=entry.level,
                pathname="<runner>",
                lineno=0,
                msg=entry.message,
                args=(),
                exc_info=None,
            )
            record.created = entry.timestamp_ms / 1000.0
            record.msecs = entry.timestamp_ms % 1000
            if entry.exception_text:
                record.exc_text = entry.exception_text

            stdlib_logging.getLogger(entry.logger_name).handle(record)

        return envelope.make_response(
            payload={"accepted": True, "count": len(batch.entries)}
        )


class PluginSupervisor:
    """插件 Supervisor

    Host 端的核心管理器，负责整个插件 Runner 进程的生命周期。
    """

    def __init__(
        self,
        plugin_dirs: Optional[List[str]] = None,
        socket_path: Optional[str] = None,
        health_check_interval_sec: float = 30.0,
    ):
        self._plugin_dirs = plugin_dirs or []
        self._health_interval = health_check_interval_sec

        # 基础设施
        self._transport = create_transport_server(socket_path=socket_path)
        self._policy = PolicyEngine()
        self._capability_service = CapabilityService(self._policy)
        self._component_registry = ComponentRegistry()
        self._event_dispatcher = EventDispatcher(self._component_registry)
        self._workflow_executor = WorkflowExecutor(self._component_registry)

        # 编解码
        from src.plugin_runtime.protocol.codec import MsgPackCodec
        codec = MsgPackCodec()

        self._rpc_server = RPCServer(
            transport=self._transport,
            codec=codec,
        )

        # Runner 子进程
        self._runner_process: Optional[asyncio.subprocess.Process] = None
        self._runner_generation: int = 0
        self._max_restart_attempts: int = 3
        self._restart_count: int = 0

        # 已注册的插件组件信息
        self._registered_plugins: Dict[str, RegisterComponentsPayload] = {}

        # 后台任务
        self._health_task: Optional[asyncio.Task] = None
        # Runner stderr 流排空任务（仅保留 stderr，用于 IPC 建立前的启动日志倒空、致命错误输出等场景）
        self._stderr_drain_task: Optional[asyncio.Task] = None
        self._running = False

        # Runner 日志桥（将 Runner 上报的批量日志重放到主进程 Logger）
        self._log_bridge: RunnerLogBridge = RunnerLogBridge()

        # 注册内部 RPC 方法
        self._register_internal_methods()

    @property
    def policy_engine(self) -> PolicyEngine:
        return self._policy

    @property
    def capability_service(self) -> CapabilityService:
        return self._capability_service

    @property
    def component_registry(self) -> ComponentRegistry:
        return self._component_registry

    @property
    def event_dispatcher(self) -> EventDispatcher:
        return self._event_dispatcher

    @property
    def workflow_executor(self) -> WorkflowExecutor:
        return self._workflow_executor

    @property
    def rpc_server(self) -> RPCServer:
        return self._rpc_server

    async def dispatch_event(
        self,
        event_type: str,
        message: Optional[Dict[str, Any]] = None,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """分发事件到所有对应 handler 的快捷方法。"""
        async def _invoke(plugin_id: str, component_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
            resp = await self.invoke_plugin(
                method="plugin.emit_event",
                plugin_id=plugin_id,
                component_name=component_name,
                args=args,
            )
            return resp.payload

        return await self._event_dispatcher.dispatch_event(
            event_type=event_type,
            invoke_fn=_invoke,
            message=message,
            extra_args=extra_args,
        )

    async def execute_workflow(
        self,
        message: Optional[Dict[str, Any]] = None,
        stream_id: Optional[str] = None,
        context: Optional[WorkflowContext] = None,
    ) -> Tuple[WorkflowResult, Optional[Dict[str, Any]], WorkflowContext]:
        """执行 Workflow Pipeline 的快捷方法。"""
        async def _invoke(plugin_id: str, component_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
            resp = await self.invoke_plugin(
                method="plugin.invoke_workflow_step",
                plugin_id=plugin_id,
                component_name=component_name,
                args=args,
            )
            payload = resp.payload
            if payload.get("success"):
                result = payload.get("result")
                return result if isinstance(result, dict) else {}
            raise RuntimeError(payload.get("result", "workflow step invoke failed"))

        return await self._workflow_executor.execute(
            invoke_fn=_invoke,
            message=message,
            stream_id=stream_id,
            context=context,
        )

    async def start(self) -> None:
        """启动 Supervisor

        1. 启动 RPC Server
        2. 拉起 Runner 子进程
        3. 启动健康检查
        """
        self._running = True

        # 启动 RPC Server
        await self._rpc_server.start()

        # 计算预期 generation（与 reload_plugins 保持一致）
        expected_generation = self._rpc_server.runner_generation + 1

        # 拉起 Runner 进程
        await self._spawn_runner()

        # 等待 Runner 完成连接，避免 start() 返回时 Runner 尚未就绪
        try:
            await self._wait_for_runner_generation(expected_generation, timeout_sec=30.0)
        except TimeoutError:
            if not self._rpc_server.is_connected:
                logger.warning("Runner 未在 30s 内完成连接，后续操作可能失败")

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
        args: Optional[Dict[str, Any]] = None,
        timeout_ms: int = 30000,
    ) -> Envelope:
        """调用插件组件

        由主进程业务逻辑调用，通过 RPC 转发给 Runner。
        """
        return await self._rpc_server.send_request(
            method=method,
            plugin_id=plugin_id,
            payload={
                "component_name": component_name,
                "args": args or {},
            },
            timeout_ms=timeout_ms,
        )

    async def reload_plugins(self, reason: str = "manual") -> None:
        """热重载所有插件（进程级 generation 切换）

        1. 拉起新 Runner
        2. 等待新 Runner 完成注册和健康检查
        3. 关停旧 Runner
        """
        logger.info(f"开始热重载插件，原因: {reason}")

        # 保存旧进程引用和旧 session token（回滚时需要恢复）
        old_process = self._runner_process
        old_registered_plugins = dict(self._registered_plugins)
        old_session_token = self._rpc_server.session_token
        expected_generation = self._rpc_server.runner_generation + 1

        # 重新生成 session token，防止被终止的旧 Runner 重连
        self._rpc_server.reset_session_token()

        # 注意：不在此处调用 _clear_runtime_state()。
        # 旧组件在新 Runner 完成注册前继续提供服务，避免热重载窗口期内
        # dispatch_event / execute_workflow 找不到任何组件导致消息静默丢失。
        # ComponentRegistry.register_component 对同名组件是覆盖式写入，安全。

        # 拉起新 Runner
        try:
            await self._spawn_runner()
            await self._wait_for_runner_generation(expected_generation, timeout_sec=30.0)
            resp = await self._rpc_server.send_request("plugin.health", timeout_ms=5000)
            health = HealthPayload.model_validate(resp.payload)
            if not health.healthy:
                raise RPCError(ErrorCode.E_PLUGIN_CRASHED, "新 Runner 健康检查失败")
        except Exception as e:
            logger.error(f"新 Runner 健康检查失败: {e}，回滚")
            await self._terminate_process(self._runner_process, old_process)
            self._runner_process = old_process
            # 恢复旧 session token，使旧 Runner 的连接仍可正常工作
            self._rpc_server.restore_session_token(old_session_token)
            self._registered_plugins = dict(old_registered_plugins)
            self._rebuild_runtime_state()
            return

        # 新 Runner 健康且已完成组件注册，现在清理旧的幽灵组件
        # 只移除不再存在于新注册表中的旧插件组件
        for old_pid in list(old_registered_plugins.keys()):
            if old_pid not in self._registered_plugins:
                self._component_registry.remove_components_by_plugin(old_pid)
                self._policy.revoke_plugin(old_pid)

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
        # Runner 日志批量上报
        self._rpc_server.register_method("runner.log_batch", self._log_bridge.handle_log_batch)

    async def _handle_register_components(self, envelope: Envelope) -> Envelope:
        """处理插件组件注册请求"""
        try:
            reg = RegisterComponentsPayload.model_validate(envelope.payload)
        except Exception as e:
            return envelope.make_error_response(ErrorCode.E_BAD_PAYLOAD.value, str(e))

        if envelope.generation != self._rpc_server.runner_generation:
            return envelope.make_error_response(
                ErrorCode.E_GENERATION_MISMATCH.value,
                f"组件注册 generation 过期: {envelope.generation} != {self._rpc_server.runner_generation}",
            )

        # 记录注册信息
        self._registered_plugins[reg.plugin_id] = reg

        # 在策略引擎中注册插件
        self._policy.register_plugin(
            plugin_id=reg.plugin_id,
            generation=envelope.generation,
            capabilities=reg.capabilities_required or [],
        )

        # 在 ComponentRegistry 中注册组件
        self._component_registry.register_plugin_components(
            plugin_id=reg.plugin_id,
            components=[c.model_dump() for c in reg.components],
        )

        stats = self._component_registry.get_stats()
        logger.info(
            f"插件 {reg.plugin_id} v{reg.plugin_version} 注册成功，"
            f"组件数: {len(reg.components)}, 能力需求: {reg.capabilities_required}，"
            f"注册表总计: {stats}"
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
            # stdout 不捕获：Runner 的日志均通过 IPC 传㛹（RunnerIPCLogHandler）
            stdout=None,
            # stderr 捕获为 PIPE，仅用于 IPC 建立前的进程级致命错误输出
            stderr=asyncio.subprocess.PIPE,
        )

        self._attach_stderr_drain(self._runner_process)
        self._runner_generation = self._rpc_server.runner_generation
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

        await self._cleanup_stderr_drain()

    async def _health_check_loop(self) -> None:
        """周期性健康检查 + 崩溃自动重启"""
        while self._running:
            await asyncio.sleep(self._health_interval)

            # 检查 Runner 进程是否意外退出
            if self._runner_process and self._runner_process.returncode is not None:
                exit_code = self._runner_process.returncode
                logger.warning(f"Runner 进程已退出 (exit_code={exit_code})")

                if self._restart_count < self._max_restart_attempts:
                    self._restart_count += 1
                    logger.info(f"尝试重启 Runner ({self._restart_count}/{self._max_restart_attempts})")
                    # 清理旧的组件注册
                    for plugin_id in list(self._registered_plugins.keys()):
                        self._component_registry.remove_components_by_plugin(plugin_id)
                        self._policy.revoke_plugin(plugin_id)
                    self._registered_plugins.clear()

                    try:
                        self._clear_runtime_state()
                        # 重新生成 session token，防止旧 Runner 僵尸进程用旧 token 重连
                        self._rpc_server.reset_session_token()
                        await self._spawn_runner()
                    except Exception as e:
                        logger.error(f"Runner 重启失败: {e}", exc_info=True)
                else:
                    logger.error(f"Runner 连续崩溃 {self._max_restart_attempts} 次，停止重启")
                continue

            if not self._rpc_server.is_connected:
                logger.warning("Runner 未连接，跳过健康检查")
                continue

            try:
                resp = await self._rpc_server.send_request("plugin.health", timeout_ms=5000)
                health = HealthPayload.model_validate(resp.payload)
                if not health.healthy:
                    logger.warning(f"Runner 健康检查异常: {health}")
                else:
                    # 健康检查成功，重置重启计数
                    self._restart_count = 0
            except RPCError as e:
                logger.error(f"健康检查失败: {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"健康检查异常: {e}")

    async def _wait_for_runner_generation(self, expected_generation: int, timeout_sec: float) -> None:
        """等待指定代际的 Runner 完成连接。"""
        deadline = asyncio.get_running_loop().time() + timeout_sec
        while asyncio.get_running_loop().time() < deadline:
            if self._rpc_server.is_connected and self._rpc_server.runner_generation >= expected_generation:
                self._runner_generation = self._rpc_server.runner_generation
                return
            await asyncio.sleep(0.1)
        raise TimeoutError(f"等待 Runner generation {expected_generation} 超时")

    def _clear_runtime_state(self) -> None:
        """清空当前插件注册态。"""
        self._component_registry.clear()
        self._policy.clear()
        self._registered_plugins.clear()

    def _rebuild_runtime_state(self) -> None:
        """根据已记录的插件注册信息重建运行时状态。"""
        self._component_registry.clear()
        self._policy.clear()
        for reg in self._registered_plugins.values():
            self._policy.register_plugin(
                plugin_id=reg.plugin_id,
                generation=self._rpc_server.runner_generation,
                capabilities=reg.capabilities_required or [],
            )
            self._component_registry.register_plugin_components(
                plugin_id=reg.plugin_id,
                components=[c.model_dump() for c in reg.components],
            )

    def _attach_stderr_drain(self, process: asyncio.subprocess.Process) -> None:
        """为 Runner stderr 创建排空任务，捕获 IPC 建立前的进程级错误输出。

        stderr 中的内容通常是：
        - Runner 启动早期（握手完成之前）的日志
        - 进程级致命错误（ImportError、SyntaxError等）
        - 异常进程退出前的最后输出

        握手成功后，插件的所有日志均经由 RunnerIPCLogHandler 通过 IPC 传输。
        """
        if process.stderr is None:
            return
        task = asyncio.create_task(
            self._drain_runner_stderr(process.stderr, process.pid),
            name=f"runner_stderr_drain:{process.pid}",
        )
        self._stderr_drain_task = task
        task.add_done_callback(
            lambda done_task: None
            if self._stderr_drain_task is not done_task
            else setattr(self, "_stderr_drain_task", None)
        )

    async def _drain_runner_stderr(
        self,
        stream: asyncio.StreamReader,
        pid: int,
    ) -> None:
        """持续读取 Runner stderr 并转发到 Host Logger，防止 PIPE 锡死子进程。

        Args:
            stream: Runner 子进程的 stderr 流。
            pid:    子进程 PID，仅用于日志上下文。
        """
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                message = line.decode(errors="replace").rstrip()
                if message:
                    # 将 stderr 输出以 WARNING 级展示：
                    # 如果 Runner 正常运行，此流应当无输出；
                    # 有输出说明进程级错误发生，需要出现在主进程日志中
                    logger.warning(f"[runner:{pid}:stderr] {message}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug(f"读取 Runner stderr 失败 (pid={pid}): {exc}")

    async def _cleanup_stderr_drain(self) -> None:
        """等待并取消 stderr 排空任务。"""
        if self._stderr_drain_task is None:
            return
        task = self._stderr_drain_task
        self._stderr_drain_task = None
        if not task.done():
            task.cancel()
        with contextlib.suppress(Exception):
            await asyncio.gather(task, return_exceptions=True)

    @staticmethod
    async def _terminate_process(
        process: Optional[asyncio.subprocess.Process],
        keep_process: Optional[asyncio.subprocess.Process] = None,
    ) -> None:
        """终止指定进程，但跳过需要保留的旧进程引用。"""
        if process is None or process is keep_process or process.returncode is not None:
            return

        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
