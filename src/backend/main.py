"""
AetherAgent FastAPI Application Entry Point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from src.backend.api.auth import router as auth_router
from src.backend.api.chat import router as chat_router
from src.backend.api.hardware import router as hardware_router
from src.backend.database import init_db

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix=settings.api_prefix)
app.include_router(chat_router, prefix=settings.api_prefix)
app.include_router(hardware_router, prefix=settings.api_prefix)


@app.on_event("startup")
async def startup():
    print(f"🚀 {settings.app_name} v{settings.app_version} starting...")
    settings.ensure_directories()
    await init_db()
    print("✅ Database initialized")
    print(f"📡 API at http://localhost:8000{settings.api_prefix}")
    print(f"📖 Docs at http://localhost:8000/docs")


@app.on_event("shutdown")
async def shutdown():
    print(f"🔒 {settings.app_name} shutting down...")


@app.get("/", tags=["Root"])
async def root():
    return {"name": settings.app_name, "version": settings.app_version, "status": "running", "docs": "/docs"}


@app.get("/health", tags=["Root"])
async def health():
    return {"status": "healthy"}
