"""Integration tests use the Compose Postgres, inside a throwaway schema that is created
before each test and dropped after it, so the real synced data is never touched.
They are skipped when that database isn't running."""

import os
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from capacity_radar.db.engine import create_session_factory, init_db

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+asyncpg://radar:radar@localhost:5432/radar"
)


@pytest.fixture
async def session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    # Every connection looks only in the radar_test schema, so tables are created there.
    engine = create_async_engine(
        TEST_DATABASE_URL, connect_args={"server_settings": {"search_path": "radar_test"}}
    )
    try:
        async with engine.begin() as conn:
            await conn.execute(text("DROP SCHEMA IF EXISTS radar_test CASCADE"))
            await conn.execute(text("CREATE SCHEMA radar_test"))
        await init_db(engine)
    except (OSError, SQLAlchemyError) as exc:
        await engine.dispose()
        pytest.skip(f"Postgres not reachable ({type(exc).__name__}); run docker compose up -d")

    yield create_session_factory(engine)

    async with engine.begin() as conn:
        await conn.execute(text("DROP SCHEMA radar_test CASCADE"))
    await engine.dispose()
