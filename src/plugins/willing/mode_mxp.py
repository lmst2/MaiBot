"""
Mxp 模式（MxpWillingManager）是一个基于意愿值动态调整的回复管理器。
其主要特点包括：
1. 通过消息频率、提及、兴趣度等多种因素动态调整回复意愿值。
2. 支持意愿值的衰减机制，确保意愿值不会无限增长。
3. 提供对单聊、群聊等不同场景的特殊处理。
4. 结合关系等级、表情包惩罚等多种参数，计算最终的回复概率。
5. 支持异步任务，用于定期更新意愿值和基础意愿值。

该模式适用于需要精细化控制回复行为的场景，能够根据用户行为和聊天环境动态调整回复策略。
"""
from .willing_manager import BaseWillingManager
from typing import Dict
import asyncio
import time
import math

class MxpWillingManager(BaseWillingManager):
    """Mxp意愿管理器"""
    def __init__(self):
        super().__init__()
        self.chat_person_reply_willing: Dict[str, Dict[str, float]] = {}  # chat_id: {person_id: 意愿值}
        self.chat_new_message_time: Dict[str, list[float]] = {}  # 聊天流ID: 消息时间
        self.last_response_person: list = ["", "", 0]  # 上次回复的用户信息

        # 可变参数
        self.intention_decay_rate = 0.93  # 意愿衰减率
        self.message_expiration_time = 120  # 消息过期时间（秒）
        self.number_of_message_storage = 10  # 消息存储数量
        self.basic_maximum_willing = 0.5  # 基础最大意愿值
        self.mention_willing_gain = 0.6  # 提及意愿增益
        self.interest_willing_gain = 0.3  # 兴趣意愿增益
        self.emoji_response_penalty = self.global_config.emoji_response_penalty # 表情包回复惩罚
        self.down_frequency_rate = self.global_config.down_frequency_rate # 降低回复频率的群组惩罚系数
        self.single_chat_gain = 0.15  # 单聊增益

    async def async_task_starter(self) -> None:
        """异步任务启动器"""
        asyncio.create_task(self._return_to_basic_willing())
        asyncio.create_task(self._chat_new_message_to_change_basic_willing())

    async def before_generate_reply_handle(self, message_id: str):
        """回复前处理"""
        pass

    async def after_generate_reply_handle(self, message_id: str):
        """回复后处理"""
        async with self.lock:
            w_info = self.ongoing_messages[message_id]
            rel_value = await w_info.person_info_manager.get_value(w_info.person_id, "relationship_value")
            rel_level = self._get_relationship_level_num(rel_value)
            self.chat_person_reply_willing[w_info.chat_id][w_info.person_id] += rel_level * 0.05

            if (w_info.chat_id == self.last_response_person[0] 
                and w_info.person_id == self.last_response_person[1] 
                and self.last_response_person[2] < 3):
                self.last_response_person[2] += 1
            else:
                self.last_response_person = [w_info.chat_id, w_info.person_id, 0]

    async def not_reply_handle(self, message_id: str):
        """不回复处理"""
        async with self.lock:
            w_info = self.ongoing_messages[message_id]
            if w_info.is_mentioned_bot:
                self.chat_person_reply_willing[w_info.chat_id][w_info.person_id] += 0.2
            if self.last_response_person[0] == w_info.chat_id and self.last_response_person[1] == w_info.person_id:
                self.chat_person_reply_willing[w_info.chat_id][w_info.person_id] += self.single_chat_gain *self.last_response_person[2]

    async def get_reply_probability(self, message_id: str):
        """获取回复概率"""
        async with self.lock:
            w_info = self.ongoing_messages[message_id]
            current_willing = self.chat_person_reply_willing[w_info.chat_id][w_info.person_id]

            if w_info.is_mentioned_bot:
                current_willing += self.mention_willing_gain / (int(current_willing) + 1)

            if w_info.interested_rate > 0:
                current_willing += math.atan(w_info.interested_rate / 2) / math.pi * 2 * self.interest_willing_gain

            self.chat_person_reply_willing[w_info.chat_id][w_info.person_id] = current_willing

            rel_value = await w_info.person_info_manager.get_value(w_info.person_id, "relationship_value")
            rel_level = self._get_relationship_level_num(rel_value)
            current_willing += rel_level * 0.1

            if self.last_response_person[0] == w_info.chat_id and self.last_response_person[1] == w_info.person_id:
                current_willing += self.single_chat_gain * self.last_response_person[2]

            chat_ongoing_messages = [msg for msg in self.ongoing_messages.values() if msg.chat_id == w_info.chat_id]
            chat_person_ogoing_messages = [msg for msg in chat_ongoing_messages if msg.person_id == w_info.person_id]
            if len(chat_person_ogoing_messages) >= 2:
                current_willing = 0
            elif len(chat_ongoing_messages) == 2:
                current_willing -= 0.5
            elif len(chat_ongoing_messages) == 3:
                current_willing -= 1.5
            elif len(chat_ongoing_messages) >= 4:
                current_willing = 0

            probability = self._willing_to_probability(current_willing)

            if w_info.is_emoji:
                probability *= self.emoji_response_penalty

            if w_info.group_info and w_info.group_info.group_id in self.global_config.talk_frequency_down_groups:
                probability /= self.down_frequency_rate

            # 打印消息信息
            mes_name = w_info.chat.group_info.group_name if w_info.chat.group_info else "私聊"
            current_time = time.strftime("%H:%M:%S", time.localtime(w_info.message.message_info.time))
            self.logger.info(
                f"[{current_time}][{mes_name}]"
                f"{w_info.chat.user_info.user_nickname}:"
                f"{w_info.message.processed_plain_text}[回复意愿:{current_willing:.2f}][概率:{probability * 100:.1f}%]"
            )

            return probability

    async def bombing_buffer_message_handle(self, message_id: str):
        """炸飞消息处理"""
        async with self.lock:
            w_info = self.ongoing_messages[message_id]
            self.chat_person_reply_willing[w_info.chat_id][w_info.person_id] += 0.1

    async def _return_to_basic_willing(self):
        """使每个人的意愿恢复到chat基础意愿"""
        while True:
            await asyncio.sleep(3)
            async with self.lock:
                for chat_id, person_willing in self.chat_person_reply_willing.items():
                    for person_id, willing in person_willing.items():
                        if chat_id not in self.chat_reply_willing:
                            self.logger.debug(f"聊天流{chat_id}不存在，错误")
                            continue
                        basic_willing = self.chat_reply_willing[chat_id]
                        person_willing[person_id] = basic_willing + (willing - basic_willing) * self.intention_decay_rate

    def setup(self, message, chat, is_mentioned_bot, interested_rate):
        super().setup(message, chat, is_mentioned_bot, interested_rate)

        self.chat_reply_willing[chat.stream_id] = self.chat_reply_willing.get(chat.stream_id, self.basic_maximum_willing)
        self.chat_person_reply_willing[chat.stream_id] = self.chat_person_reply_willing.get(chat.stream_id, {})
        self.chat_person_reply_willing[chat.stream_id][self.ongoing_messages[message.message_info.message_id].person_id] = \
        self.chat_person_reply_willing[chat.stream_id].get(self.ongoing_messages[message.message_info.message_id].person_id, 
                                                           self.chat_reply_willing[chat.stream_id])

        if chat.stream_id not in self.chat_new_message_time:
            self.chat_new_message_time[chat.stream_id] = []
        self.chat_new_message_time[chat.stream_id].append(time.time())
        self.logger.info(self.chat_new_message_time[chat.stream_id])
        if len(self.chat_new_message_time[chat.stream_id]) > self.number_of_message_storage:
            self.chat_new_message_time[chat.stream_id].pop(0)

    def _willing_to_probability(self, willing: float) -> float:
        """意愿值转化为概率"""
        willing = max(0, willing)
        if willing < 2:
            probability = math.atan(willing * 2) / math.pi * 2
        else:
            probability = math.atan(willing * 4) / math.pi * 2
        return probability
    
    async def _chat_new_message_to_change_basic_willing(self):
        """聊天流新消息改变基础意愿"""
        while True:
            update_time = 20
            await asyncio.sleep(update_time)
            async with self.lock:
                for chat_id, message_times in self.chat_new_message_time.items():

                    # 清理过期消息
                    current_time = time.time()
                    message_times = [msg_time for msg_time in message_times if current_time - msg_time < self.message_expiration_time]
                    self.chat_new_message_time[chat_id] = message_times

                    if len(message_times) < self.number_of_message_storage:
                        self.chat_reply_willing[chat_id] = self.basic_maximum_willing
                        update_time = 20
                    elif len(message_times) == self.number_of_message_storage:
                        time_interval = current_time - message_times[0]
                        basic_willing = self.basic_maximum_willing * math.sqrt(time_interval / self.message_expiration_time)
                        self.chat_reply_willing[chat_id] = basic_willing
                        update_time = 17 * math.sqrt(time_interval / self.message_expiration_time) + 3
                    else:
                        self.logger.debug(f"聊天流{chat_id}消息时间数量异常，数量：{len(message_times)}")
                        self.chat_reply_willing[chat_id] = 0

    async def get_variable_parameters(self) -> Dict[str, str]:
        """获取可变参数"""
        return {
            "intention_decay_rate": "意愿衰减率",
            "message_expiration_time": "消息过期时间（秒）",
            "number_of_message_storage": "消息存储数量",
            "basic_maximum_willing": "基础最大意愿值",
            "mention_willing_gain": "提及意愿增益",
            "interest_willing_gain": "兴趣意愿增益",
            "emoji_response_penalty": "表情包回复惩罚",
            "down_frequency_rate": "降低回复频率的群组惩罚系数",
            "single_chat_gain": "单聊增益（不仅是私聊）"
        }

    async def set_variable_parameters(self, parameters: Dict[str, any]):
        """设置可变参数"""
        async with self.lock:
            for key, value in parameters.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    self.logger.debug(f"参数 {key} 已更新为 {value}")
                else:
                    self.logger.debug(f"尝试设置未知参数 {key}")
    
    def _get_relationship_level_num(self, relationship_value) -> int:
        """关系等级计算"""
        if -1000 <= relationship_value < -227:
            level_num = 0
        elif -227 <= relationship_value < -73:
            level_num = 1
        elif -73 <= relationship_value < 227:
            level_num = 2
        elif 227 <= relationship_value < 587:
            level_num = 3
        elif 587 <= relationship_value < 900:
            level_num = 4
        elif 900 <= relationship_value <= 1000:
            level_num = 5
        else:
            level_num = 5 if relationship_value > 1000 else 0
        return level_num - 2

