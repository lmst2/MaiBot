from typing import Optional

from src.common.database.database_model import ChatSession


class MaiChatSession:
    def __init__(self, session_id: str, platform: str, user_id: Optional[str] = None, group_id: Optional[str] = None):
        self.session_id = session_id
        self.platform = platform
        self.user_id = user_id
        self.group_id = group_id

        # 验证字段
        assert self.platform, "Platform must be provided"
        assert self.user_id or self.group_id, "UserID 或 GroupID 必须提供其一"

        # 其他字段初始化
        self.is_group_session = bool(self.group_id)

    @classmethod
    def from_db_instance(cls, session: ChatSession):
        return cls(
            session_id=session.session_id,
            platform=session.platform,
            user_id=session.user_id,
            group_id=session.group_id,
        )
