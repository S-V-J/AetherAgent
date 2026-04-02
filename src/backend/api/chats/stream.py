"""SSE streaming endpoint for chat responses."""

import json
import asyncio

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.backend.api.auth import get_current_user
from src.backend.database import get_session
from src.backend.models.database import Chat, Message, User

router = APIRouter(prefix="/chats/{chat_id}", tags=["Streaming"])


class ChatMessageRequest(BaseModel):
    content: str


@router.post("/stream")
async def stream_chat(
    chat_id: str,
    request: ChatMessageRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    chat_result = await session.execute(
        select(Chat).where(Chat.id == chat_id, Chat.user_id == current_user.id)
    )
    if not chat_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Chat not found")

    user_msg = Message(chat_id=chat_id, role="user", content=request.content)
    session.add(user_msg)
    await session.flush()

    assistant_msg = Message(chat_id=chat_id, role="assistant", content="")
    session.add(assistant_msg)
    await session.flush()
    assistant_id = assistant_msg.id

    async def event_generator():
        try:
            response = (
                f"**AetherAgent Placeholder**\n\n"
                f"You said: *{request.content}*\n\n"
                f"```python\nprint('Hello from AetherAgent!')\n```\n\n"
                f"SSE streaming is working. Model inference connects in Phase 2."
            )
            for char in response:
                yield f"data: {json.dumps({'type': 'token', 'content': char})}\n\n"
                await asyncio.sleep(0.01)
            yield f"data: {json.dumps({'type': 'done', 'message_id': assistant_id, 'tokens_used': len(response.split()), 'model': 'placeholder'})}\n\n"
        except asyncio.CancelledError:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Cancelled'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
