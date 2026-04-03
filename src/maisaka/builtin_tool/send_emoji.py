"""send_emoji 内置工具。"""

from typing import Any, Dict, Optional

from src.chat.emoji_system.maisaka_tool import send_emoji_for_maisaka
from src.common.logger import get_logger
from src.core.tooling import ToolExecutionContext, ToolExecutionResult, ToolInvocation, ToolSpec
from src.maisaka.context_messages import LLMContextMessage

from .context import BuiltinToolRuntimeContext

logger = get_logger("maisaka_builtin_send_emoji")


def get_tool_spec() -> ToolSpec:
    """获取 send_emoji 工具声明。"""

    return ToolSpec(
        name="send_emoji",
        brief_description="发送一个合适的表情包来辅助表达情绪。",
        detailed_description="参数说明：\n- emotion：string，可选。希望表达的情绪，例如 happy、sad、angry 等。",
        parameters_schema={
            "type": "object",
            "properties": {
                "emotion": {
                    "type": "string",
                    "description": "希望表达的情绪，例如 happy、sad、angry 等。",
                },
            },
        },
        provider_name="maisaka_builtin",
        provider_type="builtin",
    )


async def handle_tool(
    tool_ctx: BuiltinToolRuntimeContext,
    invocation: ToolInvocation,
    context: Optional[ToolExecutionContext] = None,
) -> ToolExecutionResult:
    """执行 send_emoji 内置工具。"""

    del context
    emotion = str(invocation.arguments.get("emotion") or "").strip()
    context_texts = [
        message.processed_plain_text.strip()
        for message in tool_ctx.runtime._chat_history[-5:]
        if isinstance(message, LLMContextMessage) and message.processed_plain_text.strip()
    ]
    structured_result: Dict[str, Any] = {
        "success": False,
        "message": "",
        "description": "",
        "emotion": [],
        "requested_emotion": emotion,
        "matched_emotion": "",
    }

    logger.info(f"{tool_ctx.runtime.log_prefix} 触发表情包发送工具，请求情绪={emotion!r}")

    try:
        send_result = await send_emoji_for_maisaka(
            stream_id=tool_ctx.runtime.session_id,
            requested_emotion=emotion,
            reasoning=tool_ctx.engine.last_reasoning_content,
            context_texts=context_texts,
        )
    except Exception as exc:
        logger.exception(f"{tool_ctx.runtime.log_prefix} 发送表情包时发生异常: {exc}")
        structured_result["message"] = f"发送表情包时发生异常：{exc}"
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            structured_result["message"],
            structured_content=structured_result,
        )

    structured_result["description"] = send_result.description
    structured_result["emotion"] = list(send_result.emotions)
    structured_result["matched_emotion"] = send_result.matched_emotion
    structured_result["message"] = send_result.message

    if send_result.success:
        logger.info(
            f"{tool_ctx.runtime.log_prefix} 表情包发送成功 "
            f"描述={send_result.description!r} 情绪标签={send_result.emotions} "
            f"请求情绪={emotion!r} 命中情绪={send_result.matched_emotion!r}"
        )
        tool_ctx.append_sent_emoji_to_chat_history(
            emoji_base64=send_result.emoji_base64,
            success_message=send_result.message,
        )
        structured_result["success"] = True
        return tool_ctx.build_success_result(
            invocation.tool_name,
            send_result.message,
            structured_content=structured_result,
        )

    logger.warning(
        f"{tool_ctx.runtime.log_prefix} 表情包发送失败 "
        f"请求情绪={emotion!r} 错误信息={send_result.message}"
    )
    return tool_ctx.build_failure_result(
        invocation.tool_name,
        structured_result["message"],
        structured_content=structured_result,
    )
