"""Chat export endpoint."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.backend.api.auth import get_current_user
from src.backend.database import get_session
from src.backend.models.database import Chat, Message, User

router = APIRouter(tags=["Export"])


@router.get("/{chat_id}/export")
async def export_chat(
    chat_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(Chat).where(Chat.id == chat_id, Chat.user_id == current_user.id)
    )
    chat = result.scalar_one_or_none()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    msg_result = await session.execute(
        select(Message).where(Message.chat_id == chat_id).order_by(Message.created_at)
    )
    messages = msg_result.scalars().all()

    lines = [f"# {chat.title}\n"]
    for msg in messages:
        role = "User" if msg.role == "user" else "AetherAgent"
        lines.append(f"**{role}:**\n\n{msg.content}\n\n---\n")

    return PlainTextResponse(
        content="\n".join(lines),
        media_type="text/markdown",
        headers={"Content-Disposition": f"attachment; filename={chat.title}.md"},
    )
