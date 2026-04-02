"""Chat CRUD endpoints: create, list, get, update, delete."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.backend.api.auth import get_current_user
from src.backend.database import get_session
from src.backend.models.database import Chat, Message, User

router = APIRouter(prefix="/chats", tags=["Chats"])


class CreateChatRequest(BaseModel):
    title: str = "New Chat"


class ChatResponse(BaseModel):
    id: str
    title: str
    is_archived: bool
    created_at: datetime
    updated_at: datetime
    message_count: int = 0

    class Config:
        from_attributes = True


class UpdateChatRequest(BaseModel):
    title: str | None = None
    is_archived: bool | None = None


@router.get("/", response_model=list[ChatResponse])
async def list_chats(
    include_archived: bool = Query(False),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    query = (
        select(Chat)
        .where(Chat.user_id == current_user.id)
        .order_by(desc(Chat.updated_at))
    )
    if not include_archived:
        query = query.where(Chat.is_archived == False)
    result = await session.execute(query)
    chats = result.scalars().all()
    response = []
    for chat in chats:
        count_q = select(func.count()).where(Message.chat_id == chat.id)
        count = (await session.execute(count_q)).scalar() or 0
        response.append(ChatResponse(
            id=chat.id, title=chat.title, is_archived=chat.is_archived,
            created_at=chat.created_at, updated_at=chat.updated_at, message_count=count,
        ))
    return response


@router.post("/", response_model=ChatResponse, status_code=status.HTTP_201_CREATED)
async def create_chat(
    request: CreateChatRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    chat = Chat(user_id=current_user.id, title=request.title)
    session.add(chat)
    await session.flush()
    await session.refresh(chat)
    return ChatResponse(
        id=chat.id, title=chat.title, is_archived=chat.is_archived,
        created_at=chat.created_at, updated_at=chat.updated_at, message_count=0,
    )


@router.get("/{chat_id}", response_model=ChatResponse)
async def get_chat(
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
    count_q = select(func.count()).where(Message.chat_id == chat.id)
    count = (await session.execute(count_q)).scalar() or 0
    return ChatResponse(
        id=chat.id, title=chat.title, is_archived=chat.is_archived,
        created_at=chat.created_at, updated_at=chat.updated_at, message_count=count,
    )


@router.patch("/{chat_id}", response_model=ChatResponse)
async def update_chat(
    chat_id: str,
    request: UpdateChatRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(Chat).where(Chat.id == chat_id, Chat.user_id == current_user.id)
    )
    chat = result.scalar_one_or_none()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    if request.title is not None:
        chat.title = request.title
    if request.is_archived is not None:
        chat.is_archived = request.is_archived
    await session.flush()
    await session.refresh(chat)
    count_q = select(func.count()).where(Message.chat_id == chat.id)
    count = (await session.execute(count_q)).scalar() or 0
    return ChatResponse(
        id=chat.id, title=chat.title, is_archived=chat.is_archived,
        created_at=chat.created_at, updated_at=chat.updated_at, message_count=count,
    )


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat(
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
    await session.delete(chat)
    await session.flush()
