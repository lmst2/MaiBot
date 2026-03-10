"""策略引擎

负责能力授权校验。
每个插件在 manifest 中声明能力需求，Host 启动时签发能力令牌。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class CapabilityToken:
    """能力令牌"""
    plugin_id: str
    generation: int
    capabilities: Set[str] = field(default_factory=set)


class PolicyEngine:
    """策略引擎

    管理所有插件的能力令牌，提供授权校验。
    """

    def __init__(self):
        self._tokens: Dict[str, CapabilityToken] = {}

    def register_plugin(
        self,
        plugin_id: str,
        generation: int,
        capabilities: List[str],
    ) -> CapabilityToken:
        """为插件签发能力令牌"""
        token = CapabilityToken(
            plugin_id=plugin_id,
            generation=generation,
            capabilities=set(capabilities),
        )
        self._tokens[plugin_id] = token
        return token

    def revoke_plugin(self, plugin_id: str) -> None:
        """撤销插件的能力令牌"""
        self._tokens.pop(plugin_id, None)

    def check_capability(self, plugin_id: str, capability: str) -> Tuple[bool, str]:
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

    def get_token(self, plugin_id: str) -> Optional[CapabilityToken]:
        """获取插件的能力令牌"""
        return self._tokens.get(plugin_id)

    def list_plugins(self) -> List[str]:
        """列出所有已注册的插件"""
        return list(self._tokens.keys())
