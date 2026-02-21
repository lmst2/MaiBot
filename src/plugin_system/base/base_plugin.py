from abc import abstractmethod
from typing import Any, Callable, List, Type, Tuple, Union
from .plugin_base import PluginBase

from src.common.logger import get_logger
from src.plugin_system.base.component_types import ActionInfo, CommandInfo, EventHandlerInfo, ToolInfo
from src.plugin_system.base.workflow_types import WorkflowStepInfo
from .base_action import BaseAction
from .base_command import BaseCommand
from .base_events_handler import BaseEventHandler
from .base_tool import BaseTool

logger = get_logger("base_plugin")


class BasePlugin(PluginBase):
    """基于Action和Command的插件基类

    所有上述类型的插件都应该继承这个基类，一个插件可以包含多种组件：
    - Action组件：处理聊天中的动作
    - Command组件：处理命令请求
    - 未来可扩展：Scheduler、Listener等
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def get_plugin_components(
        self,
    ) -> List[
        Union[
            Tuple[ActionInfo, Type[BaseAction]],
            Tuple[CommandInfo, Type[BaseCommand]],
            Tuple[EventHandlerInfo, Type[BaseEventHandler]],
            Tuple[ToolInfo, Type[BaseTool]],
        ]
    ]:
        """获取插件包含的组件列表

        子类必须实现此方法，返回组件信息和组件类的列表

        Returns:
            List[tuple[ComponentInfo, Type]]: [(组件信息, 组件类), ...]
        """
        raise NotImplementedError("Subclasses must implement this method")

    def get_workflow_steps(self) -> List[Tuple[WorkflowStepInfo, Callable[..., Any]]]:
        """获取插件包含的workflow steps。

        默认返回空列表，子类可按需覆盖。
        """
        return []

    def register_plugin(self) -> bool:
        """注册插件及其所有组件"""
        from src.plugin_system.core.component_registry import component_registry

        components = self.get_plugin_components()

        # 检查依赖
        if not self._check_dependencies():
            logger.error(f"{self.log_prefix} 依赖检查失败，跳过注册")
            return False

        # 注册所有组件
        registered_components = []
        for component_info, component_class in components:
            component_info.plugin_name = self.plugin_name
            if component_registry.register_component(component_info, component_class):
                registered_components.append(component_info)
            else:
                logger.warning(f"{self.log_prefix} 组件 {component_info.name} 注册失败")

        # 更新插件信息中的组件列表
        self.plugin_info.components = registered_components

        # 注册插件
        if component_registry.register_plugin(self.plugin_info):
            # 注册workflow steps（可选）
            registered_step_count = 0
            for step_info, step_handler in self.get_workflow_steps():
                if not step_info.plugin_name:
                    step_info.plugin_name = self.plugin_name
                elif step_info.plugin_name != self.plugin_name:
                    logger.warning(
                        f"{self.log_prefix} workflow step {step_info.name} 的plugin_name({step_info.plugin_name})与当前插件不一致，已覆盖为 {self.plugin_name}"
                    )
                    step_info.plugin_name = self.plugin_name

                if component_registry.register_workflow_step(step_info, step_handler):
                    registered_step_count += 1
                else:
                    logger.warning(f"{self.log_prefix} workflow step {step_info.full_name} 注册失败")

            logger.debug(f"{self.log_prefix} 插件注册成功，包含 {len(registered_components)} 个组件")
            if registered_step_count > 0:
                logger.debug(f"{self.log_prefix} workflow steps 注册成功，数量: {registered_step_count}")
            return True
        else:
            logger.error(f"{self.log_prefix} 插件注册失败")
            return False
