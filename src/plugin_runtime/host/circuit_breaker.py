"""熔断器

为每个插件提供熔断保护，连续失败超过阈值后临时禁用。
支持指数退避恢复。
"""

from enum import Enum

import time


class CircuitState(str, Enum):
    CLOSED = "closed"        # 正常工作
    OPEN = "open"            # 熔断（拒绝所有调用）
    HALF_OPEN = "half_open"  # 探测恢复


class CircuitBreaker:
    """单个插件的熔断器"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_sec: float = 30.0,
        max_recovery_timeout_sec: float = 300.0,
    ):
        self.failure_threshold = failure_threshold
        self.base_recovery_timeout = recovery_timeout_sec
        self.max_recovery_timeout = max_recovery_timeout_sec

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._consecutive_opens = 0  # 用于指数退避

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            # 检查是否可以进入半开状态
            elapsed = time.monotonic() - self._last_failure_time
            recovery_timeout = min(
                self.base_recovery_timeout * (2 ** self._consecutive_opens),
                self.max_recovery_timeout,
            )
            if elapsed >= recovery_timeout:
                self._state = CircuitState.HALF_OPEN
        return self._state

    def allow_request(self) -> bool:
        """是否允许通过请求"""
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            return True  # 允许一次试探
        return False  # OPEN 状态拒绝

    def record_success(self) -> None:
        """记录一次成功调用"""
        if self._state == CircuitState.HALF_OPEN:
            # 半开状态成功 -> 关闭熔断
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._consecutive_opens = 0
        elif self._state == CircuitState.CLOSED:
            self._failure_count = 0

    def record_failure(self) -> None:
        """记录一次失败调用"""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._state == CircuitState.HALF_OPEN:
            # 半开状态失败 -> 重新开启熔断
            self._state = CircuitState.OPEN
            self._consecutive_opens += 1
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._consecutive_opens += 1

    def reset(self) -> None:
        """重置熔断器"""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._consecutive_opens = 0


class CircuitBreakerRegistry:
    """熔断器注册表，为每个插件维护独立的熔断器"""

    def __init__(self, **default_kwargs):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._default_kwargs = default_kwargs

    def get(self, plugin_id: str) -> CircuitBreaker:
        if plugin_id not in self._breakers:
            self._breakers[plugin_id] = CircuitBreaker(**self._default_kwargs)
        return self._breakers[plugin_id]

    def remove(self, plugin_id: str) -> None:
        self._breakers.pop(plugin_id, None)

    def reset_all(self) -> None:
        for breaker in self._breakers.values():
            breaker.reset()
