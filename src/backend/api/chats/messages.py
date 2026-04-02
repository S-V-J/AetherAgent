"""Message CRUD endpoints: list, edit, delete."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.backend.api.auth import get_current_user
from src.backend.database import get_session
from src.backend.models.database import Chat, Message, User

router = APIRouter(prefix="/chats/{chat_id}", tags=["Messages"])


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    is_edited: bool
    created_at: datetime

    class Config:
        from_attributes = True


class UpdateMessageRequest(BaseModel):
    content: str


async def _verify_chat_ownership(chat_id: str, user_id: str, session: AsyncSession) -> Chat:
    result = await session.execute(
        select(Chat).where(Chat.id == chat_id, Chat.user_id == user_id)
    )
    chat = result.scalar_one_or_none()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@router.get("/messages", response_model=list[MessageResponse])
async def get_messages(
    chat_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    await _verify_chat_ownership(chat_id, current_user.id, session)
    result = await session.execute(
        select(Message).where(Message.chat_id == chat_id).order_by(Message.created_at)
    )
    return [MessageResponse(
        id=m.id, role=m.role, content=m.content,
        is_edited=m.is_edited, created_at=m.created_at,
    ) for m in result.scalars().all()]


@router.patch("/messages/{message_id}", response_model=MessageResponse)
async def update_message(
    chat_id: str,
    message_id: str,
    request: UpdateMessageRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    await _verify_chat_ownership(chat_id, current_user.id, session)
    result = await session.execute(
        select(Message).where(Message.id == message_id, Message.chat_id == chat_id)
    )
    message = result.scalar_one_or_none()
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    if not message.is_edited:
        message.original_content = message.content
    message.content = request.content
    message.is_edited = True
    await session.flush()
    await session.refresh(message)
    return MessageResponse(
        id=message.id, role=message.role, content=message.content,
        is_edited=message.is_edited, created_at=message.created_at,
    )


@router.delete("/messages/{message_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_message(
    chat_id: str,
    message_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    await _verify_chat_ownership(chat_id, current_user.id, session)
    result = await session.execute(
        select(Message).where(Message.id == message_id, Message.chat_id == chat_id)
    )
    message = result.scalar_one_or_none()
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    await session.delete(message)
    await session.flush()
