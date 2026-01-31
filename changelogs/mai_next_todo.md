# Mai NEXT Todo list
version 0.3.0 - 2026-01-11

## 配置文件设计
- [x] 使用 `toml` 作为配置文件格式
- [x] <del>合理使用注释说明当前配置作用</del>（提案）
- [x] 使用 python 方法作为配置项说明（提案）
    - [x] 取消`bot_config_template.toml`
    - [x] 取消`model_config_template.toml`
    - [ ] 取消`template_env`
- [x] 配置类中的所有原子项目应该只包含以下类型: `str`, `int`, `float`, `bool`, `list`, `dict`, `set`
    - [x] 禁止使用 `Union` 类型
    - [x] 禁止使用`tuple`类型，使用嵌套`dataclass`替代
    - [x] 复杂类型使用嵌套配置类实现
- [x] 配置类中禁止使用除了`model_post_init`的方法
- [x] 取代了部分与标准函数混淆的命名
    - [x] `id` -> `item_id`

### BotConfig 设计
- [ ] 精简了配置项，现在只有Nickname和Alias Name了（预期将判断提及移到Adapter端）

### ChatConfig
- [x] 迁移了原来在`ChatConfig`中的方法到一个单独的临时类`TempMethodsHFC`中
    - [x] _parse_range
    - [x] get_talk_value
    - [x] 其他上面两个依赖的函数已经合并到这两个函数中

### ExpressionConfig
- [x] 迁移了原来在`ExpressionConfig`中的方法到一个单独的临时类`TempMethodsExpression`中
    - [x] get_expression_config_for_chat
    - [x] 其他上面依赖的函数已经合并到这个函数中

### ModelConfig
- [x] 迁移了原来在`ModelConfig`中的方法到一个单独的临时类`TempMethodsLLMUtils`中
    - [x] get_model_info
    - [x] get_provider

## 数据库模型设计
仅保留要点说明
### General Modifications
- [x] 所有项目增加自增编号主键`id`
- [x] 统一使用了SQLModel作为基类
- [x] 复杂类型使用JSON格式存储
- [x] 所有时间戳字段统一命名为`timestamp`
### 消息模型 MaiMessage
- [x] 自增编号主键`id`
- [x] 消息元数据
    - [x] 消息id`message_id`
    - [x] 消息时间戳`time`
    - [x] 平台名`platform`
    - [x] 用户元数据
        - [x] 用户id`user_id`
        - [x] 用户昵称`user_nickname`
        - [x] 用户备注名`user_cardname`
        - [x] 用户平台`user_platform`
    - [x] 群组元数据
        - [x] 群组id`group_id`
        - [x] 群组名称`group_name`
        - [x] 群组平台`group_platform`
    - [x] 被提及/at字段
        - [x] 是否被提及`is_mentioned`
        - [x] 是否被at`is_at`
- [x] 消息内容
    - [x] 原始消息内容`raw_content`（base64编码存储）
    - [x] 处理后的纯文本内容`processed_plain_text`
    - [x] 真正放入Prompt的消息内容`display_message`
- [x] 消息内部元数据
    - [x] 聊天会话id`session_id`
    - [x] 回复的消息id`reply_to`
    - [x] 是否为表情包消息`is_emoji`
    - [x] 是否为图片消息`is_picture`
    - [x] 是否为命令消息`is_command`
    - [x] 是否为通知消息`is_notify`
- [x] 其他配置`additional_config`（JSON格式存储）

### 模型使用情况 ModelUsage
- [x] 模型相关信息
- [x] 请求相关信息
- [x] Token使用情况

### 图片数据模型
- [x] 图片元信息
    - [x] 图片哈希值`image_hash`，使用`sha256`，同时作为图片唯一ID
- [x] 表情包的情感标签`emotion`
- [x] 是否已经被注册`is_registered`
- [x] 是否被手动禁用`is_banned`
### 动作记录模型 ActionRecord
### 命令执行记录模型 CommandRecord
新增此记录
### 在线时间记录模型 OnlineTime
### 表达方式模型
### 黑话模型
- [x] 重命名`inference_content_only`为`inference_with_content_only`
### 聊天记录模型
- [x] 重命名`original_text`为`original_message`
- [x] 重命名`forget_times`为`query_forget_count`
### 细枝末节
- [ ] 统一所有的`stream_id`和`chat_id`命名为`session_id`
- [ ] 更换Hash方式为`sha256`

## 一些细枝末节的东西
- [ ] 将`stream_id`和`chat_id`统一命名为`session_id`
- [ ] 映射表
    - [ ] `platform_group_user_session_id_map` `平台_群组_用户`-`会话ID` 映射表