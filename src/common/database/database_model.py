from typing import Optional
from sqlalchemy import Column, Float, Enum as SQLEnum
from sqlmodel import SQLModel, Field
from enum import Enum
from datetime import datetime


class ModelUser(str, Enum):
    SYSTEM = "system"
    PLUGIN = "plugin"


class ImageType(str, Enum):
    EMOJI = "emoji"
    IMAGE = "image"


class MaiMessage(SQLModel, table=True):
    __tablename__ = "mai_messages"  # type: ignore
    id: Optional[int] = Field(default=None, primary_key=True)  # 自增主键

    # 消息元数据
    message_id: str = Field(index=True, max_length=255)  # 消息id
    time: float = Field(sa_column=Column(Float))  # 消息时间，单位为秒
    platform: str = Field(index=True, max_length=100)  # 顶层平台字段
    # 消息发送者信息
    user_id: str = Field(index=True, max_length=255)  # 发送者用户id
    user_nickname: str = Field(index=True, max_length=255)  # 发送者昵称
    user_cardname: Optional[str] = Field(default=None, max_length=255, nullable=True)  # 发送者备注名
    user_platform: Optional[str] = Field(default=None, max_length=100, nullable=True)  # 发送者平台
    # 群聊信息（如果有）
    group_id: Optional[str] = Field(index=True, default=None, max_length=255, nullable=True)  # 群组id
    group_name: Optional[str] = Field(default=None, max_length=255, nullable=True)  # 群组名称
    group_platform: Optional[str] = Field(default=None, max_length=100, nullable=True)  # 群组平台
    # 被提及/at字段
    is_mentioned: bool = Field(default=False)  # 被提及
    is_at: bool = Field(default=False)  # 被at

    # 消息内部元数据
    session_id: str = Field(index=True, max_length=255)  # 聊天会话id
    reply_to: Optional[str] = Field(default=None, max_length=255, nullable=True)  # 回复的消息id
    is_emoji: bool = Field(default=False)  # 是否为表情包消息
    is_picture: bool = Field(default=False)  # 是否为图片消息
    is_command: bool = Field(default=False)  # 是否为命令
    is_notify: bool = Field(default=False)  # 是否为通知消息

    # 消息内容
    raw_content: str  # base64编码的原始消息内容
    processed_plain_text: str = Field(index=True)  # 平面化处理后的纯文本消息
    display_message: str  # 显示的消息内容（被放入Prompt）

    # 其他配置
    additional_config: Optional[str] = Field(default=None)  # 额外配置，JSON格式存储


class ModelUsage(SQLModel, table=True):
    __tablename__ = "llm_usage"  # type: ignore

    id: Optional[int] = Field(default=None, primary_key=True)  # 自增主键

    # 模型相关信息
    model_name: str = Field(index=True, max_length=255)  # 模型实际名称（供应商名称）
    model_assign_name: Optional[str] = Field(index=True, default=None, max_length=255)  # 模型分配名称（用户自定义名称）
    model_api_provider_name: str = Field(index=True, max_length=255)  # 模型API供应商名称

    # 请求相关信息
    endpoint: Optional[str] = Field(default=None, max_length=255, nullable=True)  # 模型API的具体endpoint
    user_type: ModelUser = Field(sa_column=Column(SQLEnum(ModelUser)), default=ModelUser.SYSTEM)  # 模型使用者类型
    request_type: str = Field(max_length=50)  # 内部请求类型，记录哪种模块使用了此模型
    time_cost: float = Field(sa_column=Column(Float))  # 本次请求耗时，单位秒
    timestamp: datetime = Field(default_factory=datetime.now, index=True)  # 请求时间戳

    # Token使用情况
    prompt_tokens: int  # 提示词令牌数
    completion_tokens: int  # 完成词令牌数
    total_tokens: int  # 总令牌数
    cost: float  # 本次请求的费用，单位元


class Images(SQLModel, table=True):
    """用于同时存储表情包和图片的数据库模型。"""

    __tablename__ = "images"  # type: ignore

    id: Optional[int] = Field(default=None, primary_key=True)  # 自增主键

    # 元信息
    image_hash: str = Field(default="", max_length=255)  # 图片哈希，使用sha256哈希值，亦作为图片唯一ID
    description: str  # 图片的描述
    full_path: str = Field(index=True, max_length=1024)  # 文件的完整路径 (包括文件名)
    image_type: ImageType = Field(sa_column=Column(SQLEnum(ImageType)), default=ImageType.EMOJI)
    """图片类型，例如 'emoji' 或 'image'"""
    emotion: Optional[str] = Field(default=None, nullable=True)  # 表情包的情感标签，逗号分隔

    query_count: int = Field(default=0)  # 被查询次数
    is_registered: bool = Field(default=False)  # 是否已经注册
    is_banned: bool = Field(default=False)  # 被手动禁用

    record_time: datetime = Field(default_factory=datetime.now, index=True)  # 记录时间（被创建的时间）
    register_time: Optional[datetime] = Field(default=None, nullable=True)  # 注册时间（被注册为可用表情包的时间）

    vlm_processed: bool = Field(default=False)  # 是否已经过VLM处理


class ActionRecord(SQLModel, table=True):
    """存储动作记录"""

    __tablename__ = "action_records"  # type: ignore

    id: Optional[int] = Field(default=None, primary_key=True)  # 自增主键

    # 元信息
    action_id: str = Field(index=True, max_length=255)  # 动作ID
    timestamp: datetime = Field(default_factory=datetime.now, index=True)  # 记录时间戳
    session_id: str = Field(index=True, max_length=255)  # 对应的 ChatSession session_id

    # 调用信息
    action_name: str = Field(max_length=255)  # 动作名称
    action_reasoning: Optional[str] = Field(default=None)  # 动作推理过程
    action_data: Optional[str] = Field(default=None)  # 动作数据，JSON格式存储

    action_builtin_prompt: Optional[str] = Field(default=None)  # 内置动作提示
    action_display_prompt: Optional[str] = Field(default=None)  # 最终输入到Prompt的内容


class CommandRecord(SQLModel, table=True):
    """记录命令执行情况"""

    __tablename__ = "command_records"  # type: ignore

    id: Optional[int] = Field(default=None, primary_key=True)  # 自增主键

    # 元信息
    timestamp: datetime = Field(default_factory=datetime.now, index=True)  # 记录时间戳
    session_id: str = Field(index=True, max_length=255)  # 对应的 ChatSession session_id

    # 调用信息
    command_name: str = Field(index=True, max_length=255)  # 命令名称
    command_data: Optional[str] = Field(default=None)  # 命令数据，JSON格式存储
    command_result: Optional[str] = Field(default=None)  # 命令执行结果


class OnlineTime(SQLModel, table=True):
    """
    用于存储在线时长记录的模型。
    """

    __tablename__ = "online_time"  # type: ignore

    id: Optional[int] = Field(default=None, primary_key=True)  # 自增主键

    timestamp: datetime = Field(default_factory=datetime.now, index=True)  # 时间戳
    duration_minutes: int = Field()  # 时长，单位秒
    start_timestamp: datetime = Field(default_factory=datetime.now)  # 上线时间
    end_timestamp: datetime = Field(index=True)  # 下线时间


class Expression(SQLModel, table=True):
    """用于存储表达方式的模型"""

    __tablename__ = "expressions"  # type: ignore

    id: Optional[int] = Field(default=None, primary_key=True)  # 自增主键

    situation: str = Field(index=True, max_length=255, primary_key=True)  # 情景
    style: str = Field(index=True, max_length=255, primary_key=True)  # 风格

    context: str  # 上下文
    up_content: str

    content_list: str  # 内容列表，JSON格式存储
    count: int = Field(default=0)  # 使用次数
    last_active_time: datetime = Field(default_factory=datetime.now, index=True)  # 上次使用时间
    create_time: datetime = Field(default_factory=datetime.now)  # 创建时间
    session_id: Optional[str] = Field(default=None, max_length=255, nullable=True)  # 会话ID，区分是否为全局表达方式


class Jargon(SQLModel, table=True):
    """存黑话的模型"""

    __tablename__ = "jargons"  # type: ignore

    id: Optional[int] = Field(default=None, primary_key=True)  # 自增主键

    content: str = Field(index=True, max_length=255, primary_key=True)  # 黑话内容
    raw_content: Optional[str] = Field(default=None, nullable=True)  # 原始内容，未处理的黑话内容

    meaning: str  # 黑话含义
    session_id: Optional[str] = Field(default=None, max_length=255, nullable=True)  # 会话ID，区分是否为全局黑话

    count: int = Field(default=0)  # 使用次数
    is_jargon: Optional[bool] = Field(default=True)  # 是否为黑话，False表示为白话
    is_complete: bool = Field(default=False)  # 是否为已经完成全部推断（count > 100后不再推断）
    inference_with_context: Optional[str] = Field(default=None, nullable=True)  # 带上下文的推断结果，JSON格式
    inference_with_content_only: Optional[str] = Field(default=None, nullable=True)  # 只基于词条的推断结果，JSON格式


class ChatHistory(SQLModel, table=True):
    """存储聊天历史记录的模型"""

    __tablename__ = "chat_history"  # type: ignore

    id: Optional[int] = Field(default=None, primary_key=True)  # 自增主键

    # 元信息
    session_id: str = Field(index=True, max_length=255)  # 聊天会话ID
    start_timestamp: datetime = Field(index=True)  # 聊天开始时间
    end_timestamp: datetime = Field(index=True)  # 聊天结束时间
    query_count: int = Field(default=0)  # 被检索次数
    query_forget_count: int = Field(default=0)  # 被遗忘检查的次数

    # 历史消息内容
    original_messages: str  # 对话原文
    participants: str  # 参与者列表，JSON格式存储
    theme: str  # 对话主题：这段对话的主要内容，一个简短的标题
    keywords: str  # 关键词：这段对话的关键词，JSON格式存储
    summary: str  # 概括：对这段话的平文本概括


class ThinkingQuestion(SQLModel, table=True):
    """存储思考型问题的模型"""

    __tablename__ = "thinking_questions"  # type: ignore

    id: Optional[int] = Field(default=None, primary_key=True)  # 自增主键

    # 问答对
    question: str  # 问题内容
    context: Optional[str] = Field(default=None, nullable=True)  # 上下文
    found_answer: bool = Field(default=False)  # 是否找到答案
    answer: Optional[str] = Field(default=None, nullable=True)  # 问题答案
    
    thinking_steps: Optional[str] = Field(default=None, nullable=True)  # 思考步骤，JSON格式存储
    created_timestamp: datetime = Field(default_factory=datetime.now, index=True)  # 创建时间
    updated_timestamp: datetime = Field(default_factory=datetime.now, index=True)  # 最后更新时间
