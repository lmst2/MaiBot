"""
MaiSaka built-in tool definitions.
"""

from typing import List

from src.llm_models.payload_content.tool_option import ToolOption, ToolParamType


def create_builtin_tools() -> List[ToolOption]:
    """Create built-in tools exposed to the main chat-loop model."""
    from src.llm_models.payload_content.tool_option import ToolOptionBuilder

    tools: List[ToolOption] = []

    wait_builder = ToolOptionBuilder()
    wait_builder.set_name("wait")
    wait_builder.set_description("Pause speaking and wait for the user to provide more input.")
    wait_builder.add_param(
        name="seconds",
        param_type=ToolParamType.INTEGER,
        description="How many seconds to wait before timing out.",
        required=True,
        enum_values=None,
    )
    tools.append(wait_builder.build())

    reply_builder = ToolOptionBuilder()
    reply_builder.set_name("reply")
    reply_builder.set_description("Generate and emit a visible reply based on the current thought. You must specify the target user message_id to reply to.")
    reply_builder.add_param(
        name="message_id",
        param_type=ToolParamType.STRING,
        description="The message_id of the specific user message that this reply should target.",
        required=True,
        enum_values=None,
    )
    tools.append(reply_builder.build())

    no_reply_builder = ToolOptionBuilder()
    no_reply_builder.set_name("no_reply")
    no_reply_builder.set_description("Do not emit a visible reply this round and continue thinking.")
    tools.append(no_reply_builder.build())

    stop_builder = ToolOptionBuilder()
    stop_builder.set_name("stop")
    stop_builder.set_description("Stop the current inner loop and return control to the outer chat flow.")
    tools.append(stop_builder.build())

    return tools


def get_builtin_tools() -> List[ToolOption]:
    """Return built-in tools."""
    return create_builtin_tools()
