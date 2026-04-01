"""Maisaka 内置工具声明。"""

from copy import deepcopy
from typing import Any, Dict, List

from src.core.tooling import ToolSpec, build_tool_detailed_description
from src.llm_models.payload_content.tool_option import ToolDefinitionInput


def _build_tool_spec(
    name: str,
    brief_description: str,
    parameters_schema: Dict[str, Any] | None = None,
    detailed_description: str = "",
) -> ToolSpec:
    """构建单个内置工具声明。

    Args:
        name: 工具名称。
        brief_description: 简要描述。
        parameters_schema: 参数 Schema。
        detailed_description: 详细描述；为空时自动根据参数生成。

    Returns:
        ToolSpec: 构建完成的工具声明。
    """

    normalized_schema = deepcopy(parameters_schema) if parameters_schema is not None else None
    return ToolSpec(
        name=name,
        brief_description=brief_description,
        detailed_description=(
            detailed_description.strip()
            or build_tool_detailed_description(normalized_schema)
        ),
        parameters_schema=normalized_schema,
        provider_name="maisaka_builtin",
        provider_type="builtin",
    )


def create_builtin_tool_specs() -> List[ToolSpec]:
    """创建 Maisaka 内置工具声明列表。

    Returns:
        List[ToolSpec]: 内置工具声明列表。
    """

    return [
        _build_tool_spec(
            name="wait",
            brief_description="暂停当前对话并等待用户新的输入。",
            parameters_schema={
                "type": "object",
                "properties": {
                    "seconds": {
                        "type": "integer",
                        "description": "等待的秒数。",
                    },
                },
                "required": ["seconds"],
            },
        ),
        _build_tool_spec(
            name="reply",
            brief_description="根据当前思考生成并发送一条可见回复。",
            parameters_schema={
                "type": "object",
                "properties": {
                    "msg_id": {
                        "type": "string",
                        "description": "要回复的目标用户消息编号。",
                    },
                    "quote": {
                        "type": "boolean",
                        "description": "当有非常明确的回复目标时，以引用回复的方式发送。",
                        "default": True,
                    },
                    "unknown_words": {
                        "type": "array",
                        "description": "回复前可能需要查询的黑话或词条列表。",
                        "items": {"type": "string"},
                    },
                },
                "required": ["msg_id"],
            },
        ),
        _build_tool_spec(
            name="query_jargon",
            brief_description="查询当前聊天上下文中的黑话或词条含义。",
            parameters_schema={
                "type": "object",
                "properties": {
                    "words": {
                        "type": "array",
                        "description": "要查询的词条列表。",
                        "items": {"type": "string"},
                    },
                },
                "required": ["words"],
            },
        ),
        # _build_tool_spec(
        #     name="query_person_info",
        #     brief_description="查询某个人的档案和相关记忆信息。",
        #     parameters_schema={
        #         "type": "object",
        #         "properties": {
        #             "person_name": {
        #                 "type": "string",
        #                 "description": "人物名称、昵称或用户 ID。",
        #             },
        #             "limit": {
        #                 "type": "integer",
        #                 "description": "最多返回多少条匹配记录。",
        #                 "default": 3,
        #             },
        #         },
        #         "required": ["person_name"],
        #     },
        # ),
        _build_tool_spec(
            name="no_reply",
            brief_description="本轮不进行回复，等待其他用户的新消息。",
        ),
        _build_tool_spec(
            name="send_emoji",
            brief_description="发送一个合适的表情包来辅助表达情绪。",
            parameters_schema={
                "type": "object",
                "properties": {
                    "emotion": {
                        "type": "string",
                        "description": "希望表达的情绪，例如 happy、sad、angry 等。",
                    },
                },
            },
        ),
    ]


def get_builtin_tool_specs() -> List[ToolSpec]:
    """获取 Maisaka 内置工具声明。

    Returns:
        List[ToolSpec]: 内置工具声明列表。
    """

    return create_builtin_tool_specs()


def get_builtin_tools() -> List[ToolDefinitionInput]:
    """获取兼容旧模型层的内置工具定义。

    Returns:
        List[ToolDefinitionInput]: 可直接传给模型层的工具定义。
    """

    return [tool_spec.to_llm_definition() for tool_spec in create_builtin_tool_specs()]
