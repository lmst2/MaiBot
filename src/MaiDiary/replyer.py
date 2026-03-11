"""
MaiSaka - Reply 回复生成器
根据想法和上下文生成口语化回复。
"""

from typing import Optional
from datetime import datetime
from prompt_loader import load_prompt
from llm_service import BaseLLMService
from llm_service.utils import format_chat_history


class Replyer:
    """
    回复生成器。

    根据给定的想法（reason）和对话上下文，生成符合人设的口语化回复。
    """

    def __init__(self, llm_service: Optional[BaseLLMService] = None):
        """
        初始化回复器。

        Args:
            llm_service: LLM 服务实例，如果为 None 则需要在调用前设置
        """
        self._llm_service = llm_service
        self._enabled = True

    def set_llm_service(self, llm_service: BaseLLMService) -> None:
        """设置 LLM 服务"""
        self._llm_service = llm_service

    def set_enabled(self, enabled: bool) -> None:
        """启用/禁用回复功能"""
        self._enabled = enabled

    async def reply(self, reason: str, chat_history: list) -> str:
        """
        根据想法和上下文生成回复。

        Args:
            reason: 想要回复的方式、想法、内容（不包含具体回复内容）
            chat_history: 对话历史上下文

        Returns:
            生成的回复内容，失败时返回默认回复
        """
        if not self._enabled or not reason or self._llm_service is None:
            return "..."

        # 获取当前时间
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 格式化对话历史（过滤掉 system 消息，保留其他内容）
        filtered_history = [
            msg for msg in chat_history
            if msg.get("role") != "system" and msg.get("_type") != "perception"
        ]
        formatted_history = format_chat_history(filtered_history)

        # 构建回复消息
        messages = [
            {"role": "system", "content": load_prompt("replyer.system")},
            {
                "role": "user",
                "content": (
                    f"当前时间：{current_time}\n\n"
                    f"【聊天记录】\n{formatted_history}\n\n"
                    f"【你的想法】\n{reason}\n\n"
                    f"现在，你说："
                ),
            },
        ]

        try:
            # 调用 LLM 生成回复
            from llm_service.openai_impl import OpenAILLMService
            if isinstance(self._llm_service, OpenAILLMService):
                extra_body = self._llm_service._build_extra_body()
                response = await self._llm_service._call_llm(
                    "回复生成",
                    messages,
                    temperature=0.8,
                    max_tokens=512,
                    **({"extra_body": extra_body} if extra_body else {}),
                )
                result = response.choices[0].message.content or "..."
                return result.strip()
        except Exception:
            pass

        # 生成失败时返回默认回复
        return "..."
