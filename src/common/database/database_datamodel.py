from typing import Optional
from pydantic import BaseModel
from datetime import datetime
from .database_model import ModelUser, ImageType


class MaiMessage(BaseModel):
    id: Optional[int] = None
    """自增主键"""
    message_id: str
    """消息id"""
    time: float
    """消息时间，单位为秒"""
    platform: str
    """顶层平台字段"""
    user_id: str
    """发送者用户id"""
    user_nickname: str
    """发送者昵称"""
    user_cardname: Optional[str] = None
    """发送者备注名"""
    user_platform: Optional[str] = None
    """发送者平台"""
    group_id: Optional[str] = None
    """群组id"""
    group_name: Optional[str] = None
    """群组名称"""
    group_platform: Optional[str] = None
    """群组平台"""
    is_mentioned: bool = False
    """被提及"""
    is_at: bool = False
    """被at"""
    session_id: str
    """聊天会话id"""
    reply_to: Optional[str] = None
    """回复的消息id"""
    is_emoji: bool = False
    """是否为表情包消息"""
    is_picture: bool = False
    """是否为图片消息"""
    is_command: bool = False
    """是否为命令"""
    is_notify: bool = False
    """是否为通知消息"""
    raw_content: str
    """base64编码的原始消息内容"""
    processed_plain_text: str
    """平面化处理后的纯文本消息"""
    display_message: str
    """显示的消息内容（被放入Prompt）"""
    additional_config: Optional[str] = None
    """额外配置，JSON格式存储"""


class ModelUsage(BaseModel):
    id: Optional[int] = None
    """自增主键"""
    model_name: str
    """模型实际名称（供应商名称）"""
    model_assign_name: Optional[str] = None
    """模型分配名称（用户自定义名称）"""
    model_api_provider_name: str
    """模型API供应商名称"""
    endpoint: Optional[str] = None
    """模型API的具体endpoint"""
    user_type: ModelUser = ModelUser.SYSTEM
    """模型使用者类型"""
    request_type: str
    """内部请求类型，记录哪种模块使用了此模型"""
    time_cost: float
    """本次请求耗时，单位秒"""
    timestamp: datetime
    """请求时间戳"""
    prompt_tokens: int
    """提示词令牌数"""
    completion_tokens: int
    """完成词令牌数"""
    total_tokens: int
    """总令牌数"""
    cost: float
    """本次请求的费用，单位元"""


class Images(BaseModel):
    id: Optional[int] = None
    """自增主键"""
    image_hash: str = ""
    """图片哈希，使用sha256哈希值，亦作为图片唯一ID"""
    description: str
    """图片的描述"""
    full_path: str
    """文件的完整路径 (包括文件名)"""
    image_type: ImageType = ImageType.EMOJI
    emotion: Optional[str] = None
    """表情包的情感标签，逗号分隔"""
    query_count: int = 0
    """被查询次数"""
    is_registered: bool = False
    """是否已经注册"""
    is_banned: bool = False
    """被手动禁用"""
    record_time: datetime
    """记录时间（被创建的时间）"""
    register_time: Optional[datetime] = None
    """注册时间（被注册为可用表情包的时间）"""
    vlm_processed: bool = False
    """是否已经过VLM处理"""


class ActionRecord(BaseModel):
    id: Optional[int] = None
    """自增主键"""
    action_id: str
    """动作ID"""
    timestamp: datetime
    """记录时间戳"""
    session_id: str
    """对应的 ChatSession session_id"""
    action_name: str
    """动作名称"""
    action_reasoning: Optional[str] = None
    """动作推理过程"""
    action_data: Optional[str] = None
    """动作数据，JSON格式存储"""
    action_builtin_prompt: Optional[str] = None
    """内置动作提示"""
    action_display_prompt: Optional[str] = None
    """最终输入到Prompt的内容"""


class CommandRecord(BaseModel):
    id: Optional[int] = None
    """自增主键"""
    timestamp: datetime
    """记录时间戳"""
    session_id: str
    """对应的 ChatSession session_id"""
    command_name: str
    """命令名称"""
    command_data: Optional[str] = None
    """命令数据，JSON格式存储"""
    command_result: Optional[str] = None
    """命令执行结果"""


class OnlineTime(BaseModel):
    id: Optional[int] = None
    """自增主键"""
    timestamp: datetime
    """时间戳"""
    duration_minutes: int
    """时长，单位秒"""
    start_timestamp: datetime
    """上线时间"""
    end_timestamp: datetime
    """下线时间"""


class Expression(BaseModel):
    id: Optional[int] = None
    """自增主键"""
    situation: str
    """情景"""
    style: str
    """风格"""
    context: str
    """上下文"""
    up_content: str
    content_list: str
    """内容列表，JSON格式存储"""
    count: int = 0
    """使用次数"""
    last_active_time: datetime
    """上次使用时间"""
    create_time: datetime
    """创建时间"""
    session_id: Optional[str] = None
    """会话ID，区分是否为全局表达方式"""


class Jargon(BaseModel):
    id: Optional[int] = None
    """自增主键"""
    content: str
    """黑话内容"""
    raw_content: Optional[str] = None
    """原始内容，未处理的黑话内容"""
    meaning: str
    """黑话含义"""
    session_id: Optional[str] = None
    """会话ID，区分是否为全局黑话"""
    count: int = 0
    """使用次数"""
    is_jargon: Optional[bool] = True
    """是否为黑话，False表示为白话"""
    is_complete: bool = False
    """是否为已经完成全部推断（count > 100后不再推断）"""
    inference_with_context: Optional[str] = None
    """带上下文的推断结果，JSON格式"""
    inference_with_content_only: Optional[str] = None
    """只基于词条的推断结果，JSON格式"""


class ChatHistory(BaseModel):
    id: Optional[int] = None
    """自增主键"""
    session_id: str
    """聊天会话ID"""
    start_timestamp: datetime
    """聊天开始时间"""
    end_timestamp: datetime
    """聊天结束时间"""
    query_count: int = 0
    """被检索次数"""
    query_forget_count: int = 0
    """被遗忘检查的次数"""
    original_messages: str
    """对话原文"""
    participants: str
    """参与者列表，JSON格式存储"""
    theme: str
    """对话主题：这段对话的主要内容，一个简短的标题"""
    keywords: str
    """关键词：这段对话的关键词，JSON格式存储"""
    summary: str
    """概括：对这段话的平文本概括"""


class ThinkingQuestion(BaseModel):
    id: Optional[int] = None
    """自增主键"""
    question: str
    """问题内容"""
    context: Optional[str] = None
    """上下文"""
    found_answer: bool = False
    """是否找到答案"""
    answer: Optional[str] = None
    """问题答案"""
    thinking_steps: Optional[str] = None
    """思考步骤，JSON格式存储"""
    created_timestamp: datetime
    """创建时间"""
    updated_timestamp: datetime
    """最后更新时间"""
