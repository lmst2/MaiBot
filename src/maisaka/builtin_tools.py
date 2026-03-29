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
    reply_builder.set_description(
        "Generate and emit a visible reply based on the current thought. "
        "You must specify the target user msg_id to reply to."
    )
    reply_builder.add_param(
        name="msg_id",
        param_type=ToolParamType.STRING,
        description="The msg_id of the specific user message that this reply should target.",
        required=True,
        enum_values=None,
    )
    reply_builder.add_param(
        name="quote",
        param_type=ToolParamType.BOOLEAN,
        description="Whether the visible reply should be sent as a quoted reply to the target msg_id.",
        required=False,
        enum_values=None,
    )
    reply_builder.add_param(
        name="unknown_words",
        param_type=ToolParamType.ARRAY,
        description="Optional list of words or phrases that may need jargon lookup before replying.",
        required=False,
        enum_values=None,
        items_schema={"type": "string"},
    )
    tools.append(reply_builder.build())

    query_jargon_builder = ToolOptionBuilder()
    query_jargon_builder.set_name("query_jargon")
    query_jargon_builder.set_description(
        "Query the meanings of one or more jargon words in the current chat context."
    )
    query_jargon_builder.add_param(
        name="words",
        param_type=ToolParamType.ARRAY,
        description="A list of words or phrases to query from the jargon store.",
        required=True,
        enum_values=None,
        items_schema={"type": "string"},
    )
    tools.append(query_jargon_builder.build())

    query_person_info_builder = ToolOptionBuilder()
    query_person_info_builder.set_name("query_person_info")
    query_person_info_builder.set_description(
        "Query profile and memory information about a specific person by person name, nickname, or user ID."
    )
    query_person_info_builder.add_param(
        name="person_name",
        param_type=ToolParamType.STRING,
        description="The person's name, nickname, or user ID to search for.",
        required=True,
        enum_values=None,
    )
    query_person_info_builder.add_param(
        name="limit",
        param_type=ToolParamType.INTEGER,
        description="Maximum number of matched person records to return. Defaults to 3.",
        required=False,
        enum_values=None,
    )
    tools.append(query_person_info_builder.build())

    no_reply_builder = ToolOptionBuilder()
    no_reply_builder.set_name("no_reply")
    no_reply_builder.set_description("Do not emit a visible reply this round and continue thinking.")
    tools.append(no_reply_builder.build())

    stop_builder = ToolOptionBuilder()
    stop_builder.set_name("stop")
    stop_builder.set_description("Stop the current inner loop and return control to the outer chat flow.")
    tools.append(stop_builder.build())

    send_emoji_builder = ToolOptionBuilder()
    send_emoji_builder.set_name("send_emoji")
    send_emoji_builder.set_description(
        "Send an emoji sticker to help express emotions. "
        "You should specify the emotion type to select an appropriate emoji."
    )
    send_emoji_builder.add_param(
        name="emotion",
        param_type=ToolParamType.STRING,
        description="The emotion type for selecting an appropriate emoji (e.g., 'happy', 'sad', 'angry', 'surprised', etc.).",
        required=False,
        enum_values=None,
    )
    tools.append(send_emoji_builder.build())

    return tools


def get_builtin_tools() -> List[ToolOption]:
    """Return built-in tools."""
    return create_builtin_tools()
