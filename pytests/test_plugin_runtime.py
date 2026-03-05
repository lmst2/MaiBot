"""插件运行时框架基础测试

验证协议层、传输层、RPC 通信链路的正确性。
"""

import asyncio
import sys
import os

import pytest

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# SDK 包路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "maibot-plugin-sdk"))


# ─── 协议层测试 ───────────────────────────────────────────

class TestProtocol:
    """协议层测试"""

    def test_envelope_create_and_serialize(self):
        """Envelope 创建与序列化"""
        from src.plugin_runtime.protocol.envelope import Envelope, MessageType

        env = Envelope(
            request_id=1,
            message_type=MessageType.REQUEST,
            method="plugin.invoke_command",
            plugin_id="test_plugin",
            payload={"component_name": "greet", "args": {}},
        )

        assert env.request_id == 1
        assert env.is_request()
        assert env.method == "plugin.invoke_command"

        # 测试 make_response
        resp = env.make_response(payload={"success": True})
        assert resp.is_response()
        assert resp.request_id == 1
        assert resp.payload["success"] is True

    def test_envelope_make_error_response(self):
        """错误响应生成"""
        from src.plugin_runtime.protocol.envelope import Envelope, MessageType

        env = Envelope(
            request_id=42,
            message_type=MessageType.REQUEST,
            method="cap.request",
        )

        err_resp = env.make_error_response("E_UNAUTHORIZED", "没有权限")
        assert err_resp.error is not None
        assert err_resp.error["code"] == "E_UNAUTHORIZED"
        assert err_resp.error["message"] == "没有权限"

    def test_msgpack_codec(self):
        """MsgPack 编解码"""
        from src.plugin_runtime.protocol.codec import MsgPackCodec
        from src.plugin_runtime.protocol.envelope import Envelope, MessageType

        codec = MsgPackCodec()
        env = Envelope(
            request_id=100,
            message_type=MessageType.REQUEST,
            method="test.method",
            payload={"key": "value", "number": 42},
        )

        # 编码
        data = codec.encode_envelope(env)
        assert isinstance(data, bytes)

        # 解码
        decoded = codec.decode_envelope(data)
        assert decoded.request_id == 100
        assert decoded.method == "test.method"
        assert decoded.payload["key"] == "value"
        assert decoded.payload["number"] == 42

    def test_json_codec(self):
        """JSON 编解码"""
        from src.plugin_runtime.protocol.codec import JsonCodec
        from src.plugin_runtime.protocol.envelope import Envelope, MessageType

        codec = JsonCodec()
        env = Envelope(
            request_id=200,
            message_type=MessageType.EVENT,
            method="plugin.config_updated",
            payload={"config_version": "2.0"},
        )

        data = codec.encode_envelope(env)
        assert isinstance(data, bytes)

        decoded = codec.decode_envelope(data)
        assert decoded.request_id == 200
        assert decoded.is_event()

    def test_request_id_generator(self):
        """请求 ID 生成器单调递增"""
        from src.plugin_runtime.protocol.envelope import RequestIdGenerator

        gen = RequestIdGenerator()
        ids = [gen.next() for _ in range(100)]
        assert ids == list(range(1, 101))

    def test_error_codes(self):
        """错误码枚举"""
        from src.plugin_runtime.protocol.errors import ErrorCode, RPCError

        err = RPCError(ErrorCode.E_TIMEOUT, "请求超时")
        assert err.code == ErrorCode.E_TIMEOUT
        assert "E_TIMEOUT" in str(err)

        # 序列化/反序列化
        d = err.to_dict()
        err2 = RPCError.from_dict(d)
        assert err2.code == ErrorCode.E_TIMEOUT


# ─── 传输层测试 ───────────────────────────────────────────

class TestTransport:
    """传输层测试"""

    @pytest.mark.asyncio
    async def test_uds_connection_framing(self):
        """UDS 分帧协议测试"""
        from src.plugin_runtime.transport.uds import UDSTransportServer, UDSTransportClient

        server = UDSTransportServer()
        received = asyncio.Event()
        received_data = []

        async def handler(conn):
            data = await conn.recv_frame()
            received_data.append(data)
            await conn.send_frame(b"pong")
            received.set()

        await server.start(handler)
        address = server.get_address()

        client = UDSTransportClient(address)
        conn = await client.connect()
        await conn.send_frame(b"ping")

        # 等待服务端处理
        await asyncio.wait_for(received.wait(), timeout=5.0)
        assert received_data[0] == b"ping"

        # 接收服务端回复
        resp = await conn.recv_frame()
        assert resp == b"pong"

        await conn.close()
        await server.stop()

    @pytest.mark.asyncio
    async def test_tcp_connection_framing(self):
        """TCP 分帧协议测试"""
        from src.plugin_runtime.transport.tcp import TCPTransportServer, TCPTransportClient

        server = TCPTransportServer()
        received = asyncio.Event()
        received_data = []

        async def handler(conn):
            data = await conn.recv_frame()
            received_data.append(data)
            await conn.send_frame(b"tcp_pong")
            received.set()

        await server.start(handler)
        address = server.get_address()
        host, port = address.split(":")

        client = TCPTransportClient(host, int(port))
        conn = await client.connect()
        await conn.send_frame(b"tcp_ping")

        await asyncio.wait_for(received.wait(), timeout=5.0)
        assert received_data[0] == b"tcp_ping"

        resp = await conn.recv_frame()
        assert resp == b"tcp_pong"

        await conn.close()
        await server.stop()

    @pytest.mark.asyncio
    async def test_transport_factory(self):
        """传输工厂测试"""
        from src.plugin_runtime.transport.factory import create_transport_server, create_transport_client

        server = create_transport_server()
        assert server is not None

        # UDS 路径
        client = create_transport_client("/tmp/test.sock")
        assert client is not None

        # TCP 地址
        client = create_transport_client("127.0.0.1:9999")
        assert client is not None


# ─── Host 层测试 ──────────────────────────────────────────

class TestHost:
    """Host 端基础设施测试"""

    def test_policy_engine(self):
        """策略引擎测试"""
        from src.plugin_runtime.host.policy_engine import PolicyEngine

        engine = PolicyEngine()

        # 注册插件
        token = engine.register_plugin(
            plugin_id="test_plugin",
            generation=1,
            capabilities=["send.text", "db.query"],
            limits={"qps": 10, "burst": 20},
        )

        assert token.plugin_id == "test_plugin"
        assert "send.text" in token.capabilities

        # 能力检查
        ok, _ = engine.check_capability("test_plugin", "send.text")
        assert ok

        ok, reason = engine.check_capability("test_plugin", "llm.generate")
        assert not ok
        assert "未获授权" in reason

        # 未注册插件
        ok, reason = engine.check_capability("unknown", "send.text")
        assert not ok

    def test_circuit_breaker(self):
        """熔断器测试"""
        from src.plugin_runtime.host.circuit_breaker import CircuitBreaker, CircuitState

        breaker = CircuitBreaker(failure_threshold=3)

        # 初始状态：关闭
        assert breaker.state == CircuitState.CLOSED
        assert breaker.allow_request()

        # 连续失败
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.allow_request()  # 还没到阈值

        breaker.record_failure()  # 第3次，触发熔断
        assert breaker.state == CircuitState.OPEN
        assert not breaker.allow_request()

        # 重置
        breaker.reset()
        assert breaker.state == CircuitState.CLOSED

    def test_circuit_breaker_registry(self):
        """熔断器注册表测试"""
        from src.plugin_runtime.host.circuit_breaker import CircuitBreakerRegistry

        registry = CircuitBreakerRegistry(failure_threshold=2)

        b1 = registry.get("plugin_a")
        b2 = registry.get("plugin_b")
        assert b1 is not b2
        assert registry.get("plugin_a") is b1  # 同一个


# ─── SDK 测试 ─────────────────────────────────────────────

class TestSDK:
    """SDK 框架测试"""

    def test_component_decorators(self):
        """组件装饰器测试"""
        from maibot_sdk import MaiBotPlugin, Action, Command, Tool, EventHandler
        from maibot_sdk.types import ActivationType, EventType

        class TestPlugin(MaiBotPlugin):
            @Action("greet", activation_type=ActivationType.KEYWORD, activation_keywords=["hi"])
            async def handle_greet(self, **kwargs):
                return True, "ok"

            @Command("echo", pattern=r"^/echo")
            async def handle_echo(self, **kwargs):
                return True, "echoed", 2

            @Tool("search", parameters={"query": {"type": "string"}})
            async def handle_search(self, **kwargs):
                return {"result": "found"}

            @EventHandler("on_start", event_type=EventType.ON_START)
            async def handle_start(self, **kwargs):
                return True, False, "started"

        plugin = TestPlugin()
        components = plugin.get_components()

        assert len(components) == 4

        names = {c["name"] for c in components}
        assert "greet" in names
        assert "echo" in names
        assert "search" in names
        assert "on_start" in names

        types = {c["type"] for c in components}
        assert "action" in types
        assert "command" in types
        assert "tool" in types
        assert "event_handler" in types

    def test_plugin_context_not_initialized(self):
        """未初始化上下文时应报错"""
        from maibot_sdk import MaiBotPlugin

        plugin = MaiBotPlugin()
        with pytest.raises(RuntimeError, match="尚未初始化"):
            _ = plugin.ctx

    def test_plugin_context_injection(self):
        """上下文注入测试"""
        from maibot_sdk import MaiBotPlugin
        from maibot_sdk.context import PluginContext

        plugin = MaiBotPlugin()
        ctx = PluginContext(plugin_id="test")
        plugin._set_context(ctx)

        assert plugin.ctx.plugin_id == "test"
        assert plugin.ctx.send is not None
        assert plugin.ctx.db is not None
        assert plugin.ctx.llm is not None
        assert plugin.ctx.config is not None


# ─── 端到端集成测试 ────────────────────────────────────────

class TestE2E:
    """端到端集成测试（Host + Runner 通信）"""

    @pytest.mark.asyncio
    async def test_handshake(self):
        """Host-Runner 握手流程测试"""
        from src.plugin_runtime.protocol.codec import create_codec
        from src.plugin_runtime.protocol.envelope import Envelope, HelloPayload, HelloResponsePayload, MessageType
        from src.plugin_runtime.transport.uds import UDSTransportServer, UDSTransportClient

        import secrets
        import tempfile
        import os

        socket_path = os.path.join(tempfile.gettempdir(), f"maibot-test-{os.getpid()}.sock")
        session_token = secrets.token_hex(16)
        codec = create_codec()
        handshake_done = asyncio.Event()
        server_result = {}

        async def server_handler(conn):
            # 接收握手
            data = await conn.recv_frame()
            env = codec.decode_envelope(data)
            assert env.method == "runner.hello"

            hello = HelloPayload.model_validate(env.payload)
            assert hello.session_token == session_token

            # 发送响应
            resp_payload = HelloResponsePayload(
                accepted=True,
                host_version="1.0",
                assigned_generation=1,
            )
            resp = env.make_response(payload=resp_payload.model_dump())
            await conn.send_frame(codec.encode_envelope(resp))

            server_result["runner_id"] = hello.runner_id
            handshake_done.set()

            # 保持连接一会儿
            await asyncio.sleep(1.0)

        server = UDSTransportServer(socket_path=socket_path)
        await server.start(server_handler)

        # 客户端握手
        client = UDSTransportClient(socket_path)
        conn = await client.connect()

        hello = HelloPayload(
            runner_id="test-runner",
            sdk_version="1.0.0",
            session_token=session_token,
        )
        env = Envelope(
            request_id=1,
            message_type=MessageType.REQUEST,
            method="runner.hello",
            payload=hello.model_dump(),
        )
        await conn.send_frame(codec.encode_envelope(env))

        resp_data = await conn.recv_frame()
        resp = codec.decode_envelope(resp_data)
        resp_payload = HelloResponsePayload.model_validate(resp.payload)

        assert resp_payload.accepted
        assert resp_payload.assigned_generation == 1

        await asyncio.wait_for(handshake_done.wait(), timeout=5.0)
        assert server_result["runner_id"] == "test-runner"

        await conn.close()
        await server.stop()
