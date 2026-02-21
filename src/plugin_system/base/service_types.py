from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class PluginServiceInfo:
    """插件服务注册信息"""

    name: str
    plugin_name: str
    version: str = "1.0.0"
    description: str = ""
    enabled: bool = True
    params_schema: Dict[str, Any] = field(default_factory=dict)
    return_schema: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        return f"{self.plugin_name}.{self.name}"
