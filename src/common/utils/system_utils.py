# TODO: 这个函数的实现非常临时，后续需要替换为更完善的实现，比如直接从配置文件中读取机器人自己的 ID，或者通过 API 获取机器人自己的信息等
def is_bot_self(user_id: str, platform: str) -> bool:
    """
    判断用户 ID 是否是机器人自己

    临时方法，后续会替换为更完善的实现
    """
    return user_id == "bot_self" and platform == "test_platform"
