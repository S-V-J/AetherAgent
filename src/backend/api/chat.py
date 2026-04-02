"""
AetherAgent Chat Router.
Chat CRUD, message management, and SSE streaming endpoint.
"""

import json
import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from src.backend.api.auth import get_current_user
from src.backend.database import get_session
from src.backend.models.database import Chat, Message, User

router = APIRouter(prefix="/chats", tags=["Chats"])


# --- Pydantic Schemas ---
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


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    is_edited: bool
    created_at: datetime

    class Config:
        from_attributes = True


class ChatMessageRequest(BaseModel):
    content: str


class UpdateChatRequest(BaseModel):
    title: str | None = None
    is_archived: bool | None = None


class UpdateMessageRequest(BaseModel):
    content: str


# --- Chat CRUD ---
@router.get("", response_model=list[ChatResponse])
async def list_chats(
    include_archived: bool = Query(False),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """List all chats for the current user, ordered by most recent."""
    query = (
        select(Chat)
        .where(Chat.user_id == current_user.id)
        .order_by(desc(Chat.updated_at))
    )
    if not include_archived:
        query = query.where(Chat.is_archived == False)

    result = await session.execute(query)
    chats = result.scalars().all()

    # Count messages for each chat
    response = []
    for chat in chats:
        msg_count = await session.execute(
            select(Message).where(Message.chat_id == chat.id)
        )
        response.append(ChatResponse(
            id=chat.id,
            title=chat.title,
            is_archived=chat.is_archived,
            created_at=chat.created_at,
            updated_at=chat.updated_at,
            message_count=len(msg_count.scalars().all()),
        ))
    return response


@router.post("", response_model=ChatResponse, status_code=status.HTTP_201_CREATED)
async def create_chat(
    request: CreateChatRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Create a new chat."""
    chat = Chat(
        user_id=current_user.id,
        title=request.title,
    )
    session.add(chat)
    await session.flush()
    await session.refresh(chat)
    return ChatResponse(
        id=chat.id,
        title=chat.title,
        is_archived=chat.is_archived,
        created_at=chat.created_at,
        updated_at=chat.updated_at,
        message_count=0,
    )


@router.get("/{chat_id}", response_model=ChatResponse)
async def get_chat(
    chat_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Get a single chat by ID."""
    result = await session.execute(
        select(Chat).where(Chat.id == chat_id, Chat.user_id == current_user.id)
    )
    chat = result.scalar_one_or_none()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    msg_count = await session.execute(
        select(Message).where(Message.chat_id == chat.id)
    )
    return ChatResponse(
        id=chat.id,
        title=chat.title,
        is_archived=chat.is_archived,
        created_at=chat.created_at,
        updated_at=chat.updated_at,
        message_count=len(msg_count.scalars().all()),
    )


@router.patch("/{chat_id}", response_model=ChatResponse)
async def update_chat(
    chat_id: str,
    request: UpdateChatRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Update chat title or archive status."""
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

    msg_count = await session.execute(
        select(Message).where(Message.chat_id == chat.id)
    )
    return ChatResponse(
        id=chat.id,
        title=chat.title,
        is_archived=chat.is_archived,
        created_at=chat.created_at,
        updated_at=chat.updated_at,
        message_count=len(msg_count.scalars().all()),
    )


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat(
    chat_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Delete a chat and all its messages."""
    result = await session.execute(
        select(Chat).where(Chat.id == chat_id, Chat.user_id == current_user.id)
    )
    chat = result.scalar_one_or_none()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    await session.delete(chat)
    await session.flush()


# --- Messages ---
@router.get("/{chat_id}/messages", response_model=list[MessageResponse])
async def get_messages(
    chat_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Get all messages in a chat, ordered by creation time."""
    # Verify chat belongs to user
    chat_result = await session.execute(
        select(Chat).where(Chat.id == chat_id, Chat.user_id == current_user.id)
    )
    if not chat_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Chat not found")

    result = await session.execute(
        select(Message)
        .where(Message.chat_id == chat_id)
        .order_by(Message.created_at)
    )
    messages = result.scalars().all()
    return [MessageResponse(
        id=m.id,
        role=m.role,
        content=m.content,
        is_edited=m.is_edited,
        created_at=m.created_at,
    ) for m in messages]


@router.patch("/{chat_id}/messages/{message_id}", response_model=MessageResponse)
async def update_message(
    chat_id: str,
    message_id: str,
    request: UpdateMessageRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Edit a message (stores original content for undo)."""
    result = await session.execute(
        select(Message).where(
            Message.id == message_id,
            Message.chat_id == chat_id,
        )
    )
    message = result.scalar_one_or_none()
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")

    # Verify chat ownership
    chat_result = await session.execute(
        select(Chat).where(Chat.id == chat_id, Chat.user_id == current_user.id)
    )
    if not chat_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Chat not found")

    if not message.is_edited:
        message.original_content = message.content
    message.content = request.content
    message.is_edited = True

    await session.flush()
    await session.refresh(message)
    return MessageResponse(
        id=message.id,
        role=message.role,
        content=message.content,
        is_edited=message.is_edited,
        created_at=message.created_at,
    )


@router.delete("/{chat_id}/messages/{message_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_message(
    chat_id: str,
    message_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Delete a message."""
    result = await session.execute(
        select(Message).where(
            Message.id == message_id,
            Message.chat_id == chat_id,
        )
    )
    message = result.scalar_one_or_none()
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")

    chat_result = await session.execute(
        select(Chat).where(Chat.id == chat_id, Chat.user_id == current_user.id)
    )
    if not chat_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Chat not found")

    await session.delete(message)
    await session.flush()


# --- SSE Streaming Endpoint ---
@router.post("/{chat_id}/stream")
async def stream_chat(
    chat_id: str,
    request: ChatMessageRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """
    Send a message and receive a streaming AI response via SSE.

    SSE Event Types:
    - token: a single text token/character
    - tool_start: a tool begins executing
    - tool_complete: a tool finished
    - done: generation complete with metadata
    - error: an error occurred
    """
    # Verify chat ownership
    chat_result = await session.execute(
        select(Chat).where(Chat.id == chat_id, Chat.user_id == current_user.id)
    )
    chat = chat_result.scalar_one_or_none()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Save user message to database
    user_message = Message(
        chat_id=chat_id,
        role="user",
        content=request.content,
    )
    session.add(user_message)
    await session.flush()

    # Create placeholder assistant message
    assistant_message = Message(
        chat_id=chat_id,
        role="assistant",
        content="",
    )
    session.add(assistant_message)
    await session.flush()
    assistant_id = assistant_message.id

    async def event_generator():
        """Generate SSE events. Placeholder until model is loaded."""
        try:
            # TODO: Replace with actual inference engine call (Phase 2)
            placeholder_response = (
                f"**AetherAgent Placeholder Response**\n\n"
                f"You said: *{request.content}*\n\n"
                f"This is a streaming placeholder. The inference engine will be "
                f"connected in Phase 2. Your message has been saved to the database "
                f"(message ID: `{user_message.id}`).\n\n"
                f"Hardware: RTX 4060 8GB | Backend: llama.cpp | "
                f"Quantization: Q5_K_M\n\n"
                f"```python\n"
                f"print('Hello from AetherAgent!')\n"
                f"```\n\n"
                f"The SSE stream is working character-by-character."
            )

            # Stream character by character
            for char in placeholder_response:
                data = json.dumps({"type": "token", "content": char})
                yield f"data: {data}\n\n"
                await asyncio.sleep(0.01)  # Simulate generation speed

            # Send done event with metadata
            metadata = {
                "type": "done",
                "message_id": assistant_id,
                "tokens_used": len(placeholder_response.split()),
                "model": "placeholder",
            }
            yield f"data: {json.dumps(metadata)}\n\n"

            # Update assistant message in DB (done outside streaming via a final flush)
            # Note: In production, this would be handled by the inference callback

        except asyncio.CancelledError:
            # User disconnected (clicked Stop)
            yield f"data: {json.dumps({'type': 'error', 'message': 'Stream cancelled by user'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# --- Export Chat ---
@router.get("/{chat_id}/export")
async def export_chat(
    chat_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Export a chat as a Markdown file."""
    chat_result = await session.execute(
        select(Chat).where(Chat.id == chat_id, Chat.user_id == current_user.id)
    )
    chat = chat_result.scalar_one_or_none()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    msg_result = await session.execute(
        select(Message)
        .where(Message.chat_id == chat_id)
        .order_by(Message.created_at)
    )
    messages = msg_result.scalars().all()

    lines = [f"# {chat.title}\n"]
    for msg in messages:
        role = "User" if msg.role == "user" else "AetherAgent"
        lines.append(f"**{role}:**\n\n{msg.content}\n\n---\n")

    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(
        content="\n".join(lines),
        media_type="text/markdown",
        headers={"Content-Disposition": f"attachment; filename={chat.title}.md"},
    )
