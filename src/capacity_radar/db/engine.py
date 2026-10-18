from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from capacity_radar.db.models import Base
from capacity_radar.settings import Settings


def create_engine(settings: Settings) -> AsyncEngine:
    "Builds the connection pool"
    return create_async_engine(settings.database_url.get_secret_value(), pool_pre_ping=True)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    "Makes sessions, one per request"
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(engine: AsyncEngine) -> None:
    "Creates tables at startup"
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
