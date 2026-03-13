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

    def __init__(self) -> None:
        self._tokens: Dict[str, Dict[int, CapabilityToken]] = {}

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
        self._tokens.setdefault(plugin_id, {})[generation] = token
        return token

    def revoke_plugin(self, plugin_id: str, generation: Optional[int] = None) -> None:
        """撤销插件的能力令牌。"""
        if generation is None:
            self._tokens.pop(plugin_id, None)
            return

        generations = self._tokens.get(plugin_id)
        if generations is None:
            return

        generations.pop(generation, None)
        if not generations:
            self._tokens.pop(plugin_id, None)

    def clear(self) -> None:
        """清空所有能力令牌。"""
        self._tokens.clear()

    def check_capability(self, plugin_id: str, capability: str, generation: Optional[int] = None) -> Tuple[bool, str]:
        """检查插件是否有权调用某项能力

        Returns:
            (allowed, reason)
        """
        generations = self._tokens.get(plugin_id)
        if not generations:
            return False, f"插件 {plugin_id} 未注册能力令牌"

        if generation is None:
            token = generations[max(generations)]
        else:
            token = generations.get(generation)
            if token is None:
                active_generation = max(generations)
                return False, f"插件 {plugin_id} generation 不匹配: {generation} != {active_generation}"

        if capability not in token.capabilities:
            return False, f"插件 {plugin_id} 未获授权能力: {capability}"

        if generation is not None and token.generation != generation:
            return False, f"插件 {plugin_id} generation 不匹配: {generation} != {token.generation}"

        return True, ""

    def get_token(self, plugin_id: str) -> Optional[CapabilityToken]:
        """获取插件的能力令牌"""
        generations = self._tokens.get(plugin_id)
        if not generations:
            return None
        return generations[max(generations)]

    def list_plugins(self) -> List[str]:
        """列出所有已注册的插件"""
        return list(self._tokens.keys())
