"""
Chat API Router Aggregator.
Each sub-module owns its own prefix. This file just combines them.
To add a new chat endpoint: create a file, define router with prefix, import below.
"""
from fastapi import APIRouter

from src.backend.api.chats.crud import router as crud_router
from src.backend.api.chats.messages import router as messages_router
from src.backend.api.chats.stream import router as stream_router
from src.backend.api.chats.export import router as export_router

router = APIRouter()
router.include_router(crud_router)
router.include_router(messages_router)
router.include_router(stream_router)
router.include_router(export_router)
