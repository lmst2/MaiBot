"""回复生成结果相关数据模型。

该模块用于描述新版本回复链中的三个层次：

1. LLM 原始完成结果。
2. 生成过程中的耗时与调试信息。
3. 回复链最终返回给上层的结构化结果。
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from . import BaseDataModel

if TYPE_CHECKING:
    from src.common.data_models.message_component_data_model import MessageSequence
    from src.llm_models.payload_content.tool_option import ToolCall


@dataclass
class LLMCompletionResult(BaseDataModel):
    """一次 LLM 调用的原始完成结果。

    该模型只描述模型调用本身的输入与输出，不承载回复切分、
    消息序列拼装或表达方式选择等后处理结果。

    Attributes:
        request_prompt: 实际发送给模型的 Prompt 文本。
        response_text: 模型返回的主文本内容。
        reasoning_text: 模型返回的推理内容。
        model_name: 本次请求实际使用的模型名称。
        tool_calls: 模型返回的工具调用列表。
    """

    request_prompt: str = field(
        default_factory=str,
        metadata={"description": "实际发送给模型的 Prompt 文本。"},
    )
    response_text: str = field(
        default_factory=str,
        metadata={"description": "模型返回的主文本内容。"},
    )
    reasoning_text: str = field(
        default_factory=str,
        metadata={"description": "模型返回的推理内容。"},
    )
    model_name: str = field(
        default_factory=str,
        metadata={"description": "本次请求实际使用的模型名称。"},
    )
    tool_calls: List["ToolCall"] = field(
        default_factory=list,
        metadata={"description": "模型返回的工具调用列表。"},
    )


@dataclass
class GenerationMetrics(BaseDataModel):
    """一次生成流程的耗时与调试指标。

    Attributes:
        prompt_ms: Prompt 构建耗时，单位为毫秒。
        llm_ms: LLM 调用耗时，单位为毫秒。
        overall_ms: 整个生成流程总耗时，单位为毫秒。
        stage_logs: 各阶段的简短耗时日志列表。
        extra: 额外指标字典，用于承载不适合单独升格为字段的监控信息。
    """

    prompt_ms: Optional[float] = field(
        default=None,
        metadata={"description": "Prompt 构建耗时，单位为毫秒。"},
    )
    llm_ms: Optional[float] = field(
        default=None,
        metadata={"description": "LLM 调用耗时，单位为毫秒。"},
    )
    overall_ms: Optional[float] = field(
        default=None,
        metadata={"description": "整个生成流程总耗时，单位为毫秒。"},
    )
    stage_logs: List[str] = field(
        default_factory=list,
        metadata={"description": "各阶段的简短耗时日志列表。"},
    )
    extra: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"description": "额外指标字典，用于承载动态监控信息。"},
    )


@dataclass
class ReplyGenerationResult(BaseDataModel):
    """回复链的最终结构化结果。

    该模型用于承接回复器和生成服务合并后的最终产物，供 HFC、
    BrainChat、发送服务和日志系统继续消费。

    Attributes:
        success: 本次回复生成是否成功。
        completion: LLM 原始完成结果。
        metrics: 本次生成的耗时与调试指标。
        selected_expression_ids: 本次选中的表达方式 ID 列表。
        text_fragments: 对模型输出进行切分、规范化后的文本片段列表。
        message_sequence: 最终可直接发送的消息序列。
        error_message: 失败时的错误描述；成功时为空。
    """

    success: bool = field(
        default=False,
        metadata={"description": "本次回复生成是否成功。"},
    )
    completion: LLMCompletionResult = field(
        default_factory=LLMCompletionResult,
        metadata={"description": "一次 LLM 调用的原始完成结果。"},
    )
    metrics: GenerationMetrics = field(
        default_factory=GenerationMetrics,
        metadata={"description": "本次生成的耗时与调试指标。"},
    )
    selected_expression_ids: List[int] = field(
        default_factory=list,
        metadata={"description": "本次选中的表达方式 ID 列表。"},
    )
    text_fragments: List[str] = field(
        default_factory=list,
        metadata={"description": "对模型输出进行切分、规范化后的文本片段列表。"},
    )
    message_sequence: Optional["MessageSequence"] = field(
        default=None,
        metadata={"description": "最终可直接发送的消息序列。"},
    )
    error_message: str = field(
        default_factory=str,
        metadata={"description": "失败时的错误描述；成功时通常为空字符串。"},
    )


__all__ = [
    "GenerationMetrics",
    "LLMCompletionResult",
    "ReplyGenerationResult",
]
