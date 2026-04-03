"""reply 内置工具。"""

from typing import Optional

from src.chat.replyer.replyer_manager import replyer_manager
from src.common.logger import get_logger
from src.core.tooling import ToolExecutionContext, ToolExecutionResult, ToolInvocation, ToolSpec
from src.services import send_service

from .context import BuiltinToolRuntimeContext

logger = get_logger("maisaka_builtin_reply")


def get_tool_spec() -> ToolSpec:
    """获取 reply 工具声明。"""

    return ToolSpec(
        name="reply",
        brief_description="根据当前思考生成并发送一条可见回复。",
        detailed_description=(
            "参数说明：\n"
            "- msg_id：string，必填。要回复的目标用户消息编号。\n"
            "- quote：boolean，可选。当有非常明确的回复目标时，以引用回复的方式发送，默认 true。\n"
            "- unknown_words：array，可选。回复前可能需要查询的黑话或词条列表。"
        ),
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
        provider_name="maisaka_builtin",
        provider_type="builtin",
    )


async def handle_tool(
    tool_ctx: BuiltinToolRuntimeContext,
    invocation: ToolInvocation,
    context: Optional[ToolExecutionContext] = None,
) -> ToolExecutionResult:
    """执行 reply 内置工具。"""

    latest_thought = context.reasoning if context is not None else invocation.reasoning
    target_message_id = str(invocation.arguments.get("msg_id") or "").strip()
    quote_reply = bool(invocation.arguments.get("quote", True))
    raw_unknown_words = invocation.arguments.get("unknown_words")
    unknown_words = raw_unknown_words if isinstance(raw_unknown_words, list) else None

    if not target_message_id:
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            "回复工具需要提供有效的 `msg_id` 参数。",
        )

    target_message = tool_ctx.runtime._source_messages_by_id.get(target_message_id)
    if target_message is None:
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            f"未找到要回复的目标消息，msg_id={target_message_id}",
        )

    logger.info(
        f"{tool_ctx.runtime.log_prefix} 已触发回复工具 "
        f"目标消息编号={target_message_id} 引用回复={quote_reply} 最新思考={latest_thought!r}"
    )
    try:
        replyer = replyer_manager.get_replyer(
            chat_stream=tool_ctx.runtime.chat_stream,
            request_type="maisaka_replyer",
            replyer_type="maisaka",
        )
    except Exception:
        logger.exception(
            f"{tool_ctx.runtime.log_prefix} 获取回复生成器时发生异常: 目标消息编号={target_message_id}"
        )
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            "获取 Maisaka 回复生成器时发生异常。",
        )

    if replyer is None:
        logger.error(f"{tool_ctx.runtime.log_prefix} 获取 Maisaka 回复生成器失败")
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            "Maisaka 回复生成器当前不可用。",
        )

    try:
        success, reply_result = await replyer.generate_reply_with_context(
            reply_reason=latest_thought,
            stream_id=tool_ctx.runtime.session_id,
            reply_message=target_message,
            chat_history=tool_ctx.runtime._chat_history,
            unknown_words=unknown_words,
            log_reply=False,
        )
    except Exception as exc:
        logger.exception(
            f"{tool_ctx.runtime.log_prefix} 回复生成器执行异常: 目标消息编号={target_message_id} 异常={exc}"
        )
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            "生成可见回复时发生异常。",
        )

    reply_text = reply_result.completion.response_text.strip() if success else ""
    if not reply_text:
        logger.warning(
            f"{tool_ctx.runtime.log_prefix} 回复生成器返回空文本: "
            f"目标消息编号={target_message_id} 错误信息={reply_result.error_message!r}"
        )
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            "生成可见回复失败。",
        )

    reply_segments = tool_ctx.post_process_reply_text(reply_text)
    combined_reply_text = "".join(reply_segments)
    try:
        sent = False
        for index, segment in enumerate(reply_segments):
            sent = await send_service.text_to_stream(
                text=segment,
                stream_id=tool_ctx.runtime.session_id,
                set_reply=quote_reply if index == 0 else False,
                reply_message=target_message if quote_reply and index == 0 else None,
                selected_expressions=reply_result.selected_expression_ids or None,
                typing=index > 0,
            )
            if not sent:
                break
    except Exception:
        logger.exception(
            f"{tool_ctx.runtime.log_prefix} 发送文字消息时发生异常，目标消息编号={target_message_id}"
        )
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            "发送可见回复时发生异常。",
        )

    if not sent:
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            "可见回复生成成功，但发送失败。",
            structured_content={
                "msg_id": target_message_id,
                "quote": quote_reply,
                "reply_segments": reply_segments,
            },
        )

    target_user_info = target_message.message_info.user_info
    target_user_name = target_user_info.user_cardname or target_user_info.user_nickname or target_user_info.user_id

    tool_ctx.append_guided_reply_to_chat_history(combined_reply_text)
    return tool_ctx.build_success_result(
        invocation.tool_name,
        "回复已生成并发送。",
        structured_content={
            "msg_id": target_message_id,
            "quote": quote_reply,
            "reply_text": combined_reply_text,
            "reply_segments": reply_segments,
            "target_user_name": target_user_name,
        },
    )
