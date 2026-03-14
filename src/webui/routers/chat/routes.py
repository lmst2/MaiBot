"""本地聊天室路由 - WebUI 与麦麦直接对话。"""

import uuid

from typing import Optional

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from sqlalchemy import case, func
from sqlmodel import col, select

from src.common.database.database import get_db_session
from src.common.database.database_model import PersonInfo
from src.common.logger import get_logger
from src.config.config import global_config
from src.webui.dependencies import require_auth

from .support import (
    WEBUI_CHAT_GROUP_ID,
    WEBUI_CHAT_PLATFORM,
    authenticate_chat_websocket,
    chat_history,
    chat_manager,
    dispatch_chat_event,
    normalize_webui_user_id,
    resolve_initial_virtual_identity,
    send_initial_chat_state,
)

logger = get_logger("webui.chat")

router = APIRouter(prefix="/api/chat", tags=["LocalChat"], dependencies=[Depends(require_auth)])


@router.get("/history")
async def get_chat_history(
    limit: int = Query(default=50, ge=1, le=200),
    user_id: Optional[str] = Query(default=None),
    group_id: Optional[str] = Query(default=None),
) -> dict[str, object]:
    """获取聊天历史记录。"""
    del user_id
    target_group_id = group_id or WEBUI_CHAT_GROUP_ID
    history = chat_history.get_history(limit, target_group_id)
    return {"success": True, "messages": history, "total": len(history)}


@router.get("/platforms")
async def get_available_platforms() -> dict[str, object]:
    """获取可用平台列表。"""
    try:
        with get_db_session() as session:
            statement = (
                select(PersonInfo.platform, func.count().label("count"))
                .group_by(PersonInfo.platform)
                .order_by(func.count().desc())
            )
            platforms = session.exec(statement).all()

        result = [{"platform": platform, "count": count} for platform, count in platforms if platform]
        return {"success": True, "platforms": result}
    except Exception as e:
        logger.error(f"获取平台列表失败: {e}")
        return {"success": False, "error": str(e), "platforms": []}


@router.get("/persons")
async def get_persons_by_platform(
    platform: str = Query(..., description="平台名称"),
    search: Optional[str] = Query(default=None, description="搜索关键词"),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, object]:
    """获取指定平台的用户列表。"""
    try:
        statement = select(PersonInfo).where(col(PersonInfo.platform) == platform)
        if search:
            statement = statement.where(
                (col(PersonInfo.person_name).contains(search))
                | (col(PersonInfo.user_nickname).contains(search))
                | (col(PersonInfo.user_id).contains(search))
            )

        statement = statement.order_by(
            case((col(PersonInfo.last_known_time).is_(None), 1), else_=0),
            col(PersonInfo.last_known_time).desc(),
        ).limit(limit)

        with get_db_session() as session:
            persons = session.exec(statement).all()

        result = [
            {
                "person_id": person.person_id,
                "user_id": person.user_id,
                "person_name": person.person_name,
                "nickname": person.user_nickname,
                "is_known": person.is_known,
                "platform": person.platform,
                "display_name": person.person_name or person.user_nickname or person.user_id,
            }
            for person in persons
        ]
        return {"success": True, "persons": result, "total": len(result)}
    except Exception as e:
        logger.error(f"获取用户列表失败: {e}")
        return {"success": False, "error": str(e), "persons": []}


@router.delete("/history")
async def clear_chat_history(
    group_id: Optional[str] = Query(default=None),
) -> dict[str, object]:
    """清空聊天历史记录。"""
    deleted = chat_history.clear_history(group_id)
    return {"success": True, "message": f"已清空 {deleted} 条聊天记录"}


@router.websocket("/ws")
async def websocket_chat(
    websocket: WebSocket,
    user_id: Optional[str] = Query(default=None),
    user_name: Optional[str] = Query(default="WebUI用户"),
    platform: Optional[str] = Query(default=None),
    person_id: Optional[str] = Query(default=None),
    group_name: Optional[str] = Query(default=None),
    group_id: Optional[str] = Query(default=None),
    token: Optional[str] = Query(default=None),
) -> None:
    """WebSocket 聊天端点。"""
    if not await authenticate_chat_websocket(websocket, token):
        logger.warning("聊天 WebSocket 连接被拒绝：认证失败")
        await websocket.close(code=4001, reason="认证失败，请重新登录")
        return

    session_id = str(uuid.uuid4())
    normalized_user_id = normalize_webui_user_id(user_id)
    current_user_name = user_name or "WebUI用户"
    current_virtual_config = resolve_initial_virtual_identity(platform, person_id, group_name, group_id)

    await chat_manager.connect(websocket, session_id, normalized_user_id)
    try:
        await send_initial_chat_state(
            session_id=session_id,
            user_id=normalized_user_id,
            user_name=current_user_name,
            virtual_config=current_virtual_config,
        )

        while True:
            data = await websocket.receive_json()
            current_user_name, current_virtual_config = await dispatch_chat_event(
                session_id=session_id,
                session_id_prefix=session_id[:8],
                data=data,
                current_user_name=current_user_name,
                normalized_user_id=normalized_user_id,
                current_virtual_config=current_virtual_config,
            )
    except WebSocketDisconnect:
        logger.info(f"WebSocket 断开: session={session_id}, user={normalized_user_id}")
    except Exception as e:
        logger.error(f"WebSocket 错误: {e}")
    finally:
        chat_manager.disconnect(session_id, normalized_user_id)


@router.get("/info")
async def get_chat_info() -> dict[str, object]:
    """获取聊天室信息。"""
    return {
        "bot_name": global_config.bot.nickname,
        "platform": WEBUI_CHAT_PLATFORM,
        "group_id": WEBUI_CHAT_GROUP_ID,
        "active_sessions": len(chat_manager.active_connections),
    }