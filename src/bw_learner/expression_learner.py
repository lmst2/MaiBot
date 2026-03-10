from datetime import datetime
from sqlmodel import select
from typing import TYPE_CHECKING, List, Optional, Tuple

import asyncio
import difflib
import json

from src.llm_models.utils_model import LLMRequest
from src.config.config import model_config, global_config
from src.prompt.prompt_manager import prompt_manager
from src.common.logger import get_logger
from src.common.database.database_model import Expression
from src.common.database.database import get_db_session
from src.common.data_models.expression_data_model import MaiExpression
from src.common.utils.utils_message import MessageUtils

from .expression_utils import check_expression_suitability, parse_expression_response

if TYPE_CHECKING:
    from src.chat.message_receive.message import SessionMessage
    from .jargon_miner import JargonMiner


logger = get_logger("expressor")

# TODO: 重构完LLM相关内容后，替换成新的模型调用方式
express_learn_model = LLMRequest(model_set=model_config.model_task_config.utils, request_type="expression.learner")
summary_model = LLMRequest(model_set=model_config.model_task_config.tool_use, request_type="expression.summary")
check_model = LLMRequest(model_set=model_config.model_task_config.tool_use, request_type="expression.check")


class ExpressionLearner:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id

        # 学习锁，防止并发执行学习任务
        self._learning_lock = asyncio.Lock()

        # 消息缓存
        self._messages_cache: List["SessionMessage"] = []

    def add_messages(self, messages: List["SessionMessage"]) -> None:
        """添加消息到缓存"""
        self._messages_cache.extend(messages)

    def get_cache_size(self) -> int:
        """获取当前消息缓存的大小"""
        return len(self._messages_cache)

    async def learn(self, jargon_miner: Optional["JargonMiner"] = None):
        """学习主流程"""
        if not self._messages_cache:
            logger.debug("没有消息可供学习，跳过学习过程")
            return
        readable_message, _, _ = await MessageUtils.build_readable_message(
            self._messages_cache,
            anonymize=True,
            show_lineno=True,
            extract_pictures=True,
        )
        self._messages_cache.clear()  # 学习后清空缓存
        prompt_template = prompt_manager.get_prompt("learn_style")
        prompt_template.add_context("bot_name", global_config.bot.nickname)
        prompt_template.add_context("chat_str", readable_message)

        prompt = await prompt_manager.render_prompt(prompt_template)

        try:
            response, _ = await express_learn_model.generate_response_async(prompt, temperature=0.3)
        except Exception as e:
            logger.error(f"学习表达方式失败,模型生成出错: {e}")
            return None

        # 解析 LLM 返回的表达方式列表和黑话列表（包含来源行编号）
        expressions: List[Tuple[str, str, str]]
        jargon_entries: List[Tuple[str, str]]  # (content, source_id)
        expressions, jargon_entries = parse_expression_response(response)
        # TODO: 完成学习

        # 从缓存检查 jargon 是否出现在 message 中
    
    # ====== 黑话相关 ======
    def _check_cached_jargons_in_messages(self, jargon_miner: Optional["JargonMiner"] = None):
        if not jargon_miner:
            return []
        # TODO: 完成检测逻辑

    # ====== DB 操作相关 ======
    async def _upsert_expression_to_db(self, situation: str, style: str):
        expr, similarity = self._find_similar_expression(situation) or (None, 0)
        if expr:
            # 根据相似度决定是否使用 LLM 总结
            # 完全匹配（相似度 == 1.0）时不总结，相似匹配时总结
            use_llm_summary = similarity < 1.0
            await self._update_existing_expression(expr, situation, use_llm_summary=use_llm_summary)
            return
        # 没有找到匹配的记录，创建新记录
        self._create_expression(situation, style)

    def _create_expression(self, situation: str, style: str):
        content_list = [situation]
        try:
            with get_db_session() as db:
                new_expr = Expression(
                    situation=situation,
                    style=style,
                    content_list=json.dumps(content_list),
                    count=1,
                    session_id=self.session_id,
                    last_active_time=datetime.now(),
                )
                db.add(new_expr)
        except Exception as e:
            logger.error(f"创建表达方式失败: {e}")

    async def _update_existing_expression(self, expr: "MaiExpression", situation: str, use_llm_summary: bool = True):
        expr.content.append(situation)
        expr.count += 1
        expr.checked = False  # count 增加时重置 checked 为 False
        expr.last_active_time = datetime.now()

        if use_llm_summary:
            # 相似匹配时，使用 LLM 重新组合 situation
            new_situation = await self._compose_situation_text(expr.content)
            if new_situation:
                expr.situation = new_situation

        try:
            with get_db_session() as session:
                if expr.item_id is None:
                    raise ValueError("表达方式对象缺少 item_id，无法更新数据库记录")
                statement = select(Expression).filter_by(id=expr.item_id).limit(1)
                if db_expr := session.exec(statement).first():
                    db_expr.content_list = json.dumps(expr.content)
                    db_expr.count = expr.count
                    db_expr.checked = expr.checked
                    db_expr.last_active_time = expr.last_active_time
                    db_expr.situation = expr.situation  # 更新 situation
                    session.add(db_expr)
                else:
                    logger.warning(f"表达方式 ID {expr.item_id} 在数据库中未找到，无法更新")
        except Exception as e:
            logger.error(f"更新表达方式失败: {e}")

        # count 增加后，立即进行一次检查
        await self._check_expression(expr)

    # ====== 概括方法 ======
    async def _compose_situation_text(self, content_list: List[str]) -> Optional[str]:
        texts = [c.strip() for c in content_list if c.strip()]
        if not texts:
            return None
        description = "\n".join(f"- {s}" for s in texts[-10:])  # 只取最近10条进行概括
        prompt = (
            "请阅读以下多个聊天情境描述，并将它们概括成一句简短的话，长度不超过20个字，保留共同特点：\n"
            f"{description}\n"
            "只输出概括内容。"
        )
        try:
            summary, _ = await summary_model.generate_response_async(prompt, temperature=0.2)
            if summary := summary.strip():
                return summary
        except Exception as e:
            logger.error(f"使用 LLM 生成表达方式概括失败: {e}")
        return None

    async def _check_expression(self, expr: "MaiExpression"):
        """
        检查表达方式（在 count 增加后调用）

        Args:
            expr (MaiExpression): 要检查的表达方式对象
        """
        if not global_config.expression.expression_self_reflect:
            logger.debug("表达方式自我反思功能未启用，跳过检查")
            return

        suitable, reason, error = await check_expression_suitability(expr.situation, expr.style)
        if error:
            logger.error(f"检查表达方式时发生错误: {error}")
            return
        expr.checked = True
        expr.rejected = not suitable

        try:
            with get_db_session() as session:
                statement = select(Expression).filter_by(id=expr.item_id).limit(1)
                if db_expr := session.exec(statement).first():
                    db_expr.checked = expr.checked
                    db_expr.rejected = expr.rejected
                    session.add(db_expr)
                else:
                    logger.warning(f"表达方式 ID {expr.item_id} 在数据库中未找到，无法更新检查结果")
        except Exception as e:
            logger.error(f"更新表达方式检查结果失败: {e}")

        status = "通过" if suitable else "不通过"
        logger.info(
            f"表达方式检查完成 [ID: {expr.item_id}] - {status} | "
            f"Situation: {expr.situation[:30]}... | "
            f"Style: {expr.style[:30]}... | "
            f"Reason: {reason[:50] if reason else '无'}..."
        )

    def _find_similar_expression(
        self, situation: str, similarity_threshold: float = 0.75
    ) -> Optional[Tuple[MaiExpression, float]]:
        """在数据库中查找相似的表达方式"""
        try:
            with get_db_session() as session:
                statement = select(Expression).filter_by(session_id=self.session_id)
                expressions = session.exec(statement).all()

            best_match: Optional[Expression] = None
            best_similarity = 0.0

            for expr in expressions:
                content_list = json.loads(expr.content_list)
                for situation in content_list:
                    similarity = difflib.SequenceMatcher(None, situation, expr.situation).ratio()
                    if similarity > similarity_threshold and similarity > best_similarity:
                        best_similarity = similarity
                        best_match = expr
            if best_match:
                logger.debug(f"找到相似表达方式情景 [ID: {best_match.id}]，相似度: {best_similarity:.2f}")
                return MaiExpression.from_db_instance(best_match), best_similarity

        except Exception as e:
            logger.error(f"查找相似表达方式失败: {e}")
        return None
