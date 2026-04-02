"""
AetherAgent Database Engine.
Async SQLAlchemy engine and session factory.
"""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from config.settings import settings
from src.backend.models.database import Base


def get_engine():
    """Create async engine based on database URL."""
    db_url = settings.database_url

    if db_url.startswith("sqlite"):
        # SQLite requires special config for async + writable foreign keys
        return create_async_engine(
            db_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=settings.debug,
        )
    else:
        # PostgreSQL / other databases
        return create_async_engine(
            db_url,
            pool_size=5,
            max_overflow=10,
            echo=settings.debug,
        )


engine = get_engine()

async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db():
    """Create all tables. Called at app startup."""
    settings.ensure_directories()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    """Dependency: yield an async database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
