"""策略引擎

负责能力授权校验、限流、配额管理。
每个插件在 manifest 中声明能力需求，Host 启动时签发能力令牌。
"""

from dataclasses import dataclass, field

import time


@dataclass
class CapabilityToken:
    """能力令牌

    描述某个插件在当前会话中被授予的能力和资源限制。
    """
    plugin_id: str
    generation: int
    capabilities: set[str] = field(default_factory=set)
    qps_limit: int = 20
    burst_limit: int = 50
    daily_token_limit: int = 200000
    max_payload_kb: int = 256

    # 运行时统计
    _call_count: int = field(default=0, init=False, repr=False)
    _window_start: float = field(default_factory=time.monotonic, init=False, repr=False)
    _window_calls: int = field(default=0, init=False, repr=False)


class PolicyEngine:
    """策略引擎

    管理所有插件的能力令牌，提供授权校验与限流决策。
    """

    def __init__(self):
        # plugin_id -> CapabilityToken
        self._tokens: dict[str, CapabilityToken] = {}

    def register_plugin(
        self,
        plugin_id: str,
        generation: int,
        capabilities: list[str],
        limits: dict | None = None,
    ) -> CapabilityToken:
        """为插件签发能力令牌"""
        limits = limits or {}
        token = CapabilityToken(
            plugin_id=plugin_id,
            generation=generation,
            capabilities=set(capabilities),
            qps_limit=limits.get("qps", 20),
            burst_limit=limits.get("burst", 50),
            daily_token_limit=limits.get("daily_tokens", 200000),
            max_payload_kb=limits.get("max_payload_kb", 256),
        )
        self._tokens[plugin_id] = token
        return token

    def revoke_plugin(self, plugin_id: str) -> None:
        """撤销插件的能力令牌"""
        self._tokens.pop(plugin_id, None)

    def check_capability(self, plugin_id: str, capability: str) -> tuple[bool, str]:
        """检查插件是否有权调用某项能力

        Returns:
            (allowed, reason)
        """
        token = self._tokens.get(plugin_id)
        if token is None:
            return False, f"插件 {plugin_id} 未注册能力令牌"

        if capability not in token.capabilities:
            return False, f"插件 {plugin_id} 未获授权能力: {capability}"

        return True, ""

    def check_rate_limit(self, plugin_id: str) -> tuple[bool, str]:
        """检查插件是否超过调用频率限制（滑动窗口）

        Returns:
            (allowed, reason)
        """
        token = self._tokens.get(plugin_id)
        if token is None:
            return False, f"插件 {plugin_id} 未注册"

        now = time.monotonic()
        elapsed = now - token._window_start

        # 每秒重置窗口
        if elapsed >= 1.0:
            token._window_start = now
            token._window_calls = 0

        token._window_calls += 1

        if token._window_calls > token.burst_limit:
            return False, f"插件 {plugin_id} 超过突发限制 ({token.burst_limit}/s)"

        return True, ""

    def check_payload_size(self, plugin_id: str, payload_size_bytes: int) -> tuple[bool, str]:
        """检查 payload 大小是否在限制内"""
        token = self._tokens.get(plugin_id)
        if token is None:
            return False, f"插件 {plugin_id} 未注册"

        max_bytes = token.max_payload_kb * 1024
        if payload_size_bytes > max_bytes:
            return False, f"payload 大小 {payload_size_bytes} 超过限制 {max_bytes}"

        return True, ""

    def get_token(self, plugin_id: str) -> CapabilityToken | None:
        """获取插件的能力令牌"""
        return self._tokens.get(plugin_id)

    def list_plugins(self) -> list[str]:
        """列出所有已注册的插件"""
        return list(self._tokens.keys())
