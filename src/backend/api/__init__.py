"""
Master API Router Registry.
To add a new API module:
1. Create a new directory under api/ (e.g., api/scheduler/)
2. Create __init__.py with an aggregated router
3. Import and add it below
main.py NEVER needs to change.
"""
from fastapi import APIRouter

from src.backend.api.auth import router as auth_router
from src.backend.api.chats import router as chats_router
from src.backend.api.hardware import router as hardware_router

api_router = APIRouter()

api_router.include_router(auth_router)
api_router.include_router(chats_router)
api_router.include_router(hardware_router)
