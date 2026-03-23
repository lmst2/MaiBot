"""
MaiSaka - 内置工具定义
定义 say, wait, stop, store_context 等内置工具
使用主项目的工具格式（ToolOption + ToolParamType）
"""

from typing import Any, Dict, List
from src.llm_models.payload_content.tool_option import ToolOption, ToolParamType


# 内置工具定义
def create_builtin_tools() -> List[ToolOption]:
    """创建内置工具列表"""
    from src.llm_models.payload_content.tool_option import ToolOptionBuilder

    tools = []

    # say 工具
    send_message_builder = ToolOptionBuilder()
    send_message_builder.set_name("send_message")
    send_message_builder.set_description(
        "对用户说话。你所有想让用户看到的正式发言都必须通过此工具输出。直接输出的文本会被视为你的内心思考，用户无法阅读。reason 参数描述你想要回复的方式、想法和内容，系统会根据你的想法和对话上下文生成具体的回复。"
    )
    send_message_builder.add_param(
        name="reason",
        param_type=ToolParamType.STRING,
        description="描述你想要回复的方式、想法和内容。例如：'同意对方的看法，并分享自己的经历' 或 '礼貌地拒绝，表示现在不方便聊天'",
        required=True,
        enum_values=None,
    )
    tools.append(send_message_builder.build())

    # wait 工具
    wait_builder = ToolOptionBuilder()
    wait_builder.set_name("wait")
    wait_builder.set_description(
        "暂时结束你的发言，把话语权交给用户，等待对方说话。这就像现实对话中你说完一句话后停下来等对方回应。如果用户在等待期间说了话，你会通过工具返回结果收到内容。如果超时没有回复，你也会收到超时通知。"
    )
    wait_builder.add_param(
        name="seconds",
        param_type=ToolParamType.INTEGER,
        description="等待的秒数。建议 3-10 秒。超过这个时间用户没有回复会显示超时提示。",
        required=True,
        enum_values=None,
    )
    tools.append(wait_builder.build())

    # stop 工具
    stop_builder = ToolOptionBuilder()
    stop_builder.set_name("stop")
    stop_builder.set_description(
        "结束当前对话循环，进入待机状态，直到用户下次输入新内容时再唤醒你。当对话自然结束、用户表示不想继续聊、或连续多次等待超时用户没有回复时使用。"
    )
    tools.append(stop_builder.build())

    # store_context 工具
    store_context_builder = ToolOptionBuilder()
    store_context_builder.set_name("store_context")
    store_context_builder.set_description(
        "将指定范围的对话上下文存入记忆系统，然后从当前对话中移除这些内容。适合在对话上下文过长、话题转换、或遇到重要内容需要保存时使用。"
    )
    store_context_builder.add_param(
        name="count",
        param_type=ToolParamType.INTEGER,
        description="要保存的消息条数（从最早的对话开始计数）。建议 5-20 条。",
        required=True,
        enum_values=None,
    )
    store_context_builder.add_param(
        name="reason",
        param_type=ToolParamType.STRING,
        description="保存原因，用于后续检索。例如：'讨论了用户的工作情况' 或 '用户分享了对电影的看法'",
        required=True,
        enum_values=None,
    )
    tools.append(store_context_builder.build())

    return tools


# 为了兼容性，创建一个函数来将工具转换为 dict 格式（用于调试显示）
def builtin_tools_as_dicts() -> List[Dict[str, Any]]:
    """将内置工具转换为 dict 格式（用于调试）"""
    return [
        {
            "name": "send_message",
            "description": "对用户说话。你所有想让用户看到的正式发言都必须通过此工具输出。",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string", "description": "回复的想法和内容"}},
                "required": ["reason"],
            },
        },
        {
            "name": "wait",
            "description": "暂时结束发言，等待用户回应",
            "parameters": {
                "type": "object",
                "properties": {"seconds": {"type": "number", "description": "等待秒数"}},
                "required": ["seconds"],
            },
        },
        {
            "name": "stop",
            "description": "结束对话循环",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "store_context",
            "description": "保存对话上下文到记忆系统",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {"type": "number", "description": "保存的消息条数"},
                    "reason": {"type": "string", "description": "保存原因"},
                },
                "required": ["count", "reason"],
            },
        },
    ]


# 导出工具创建函数和列表
def get_builtin_tools() -> List[ToolOption]:
    """获取内置工具列表"""
    return create_builtin_tools()


# 为了向后兼容，也导出 dict 格式
BUILTIN_TOOLS_DICTS = builtin_tools_as_dicts()
