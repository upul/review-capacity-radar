"""FastAPI application: startup, shutdown and app construction."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from capacity_radar.api.errors import register_error_handlers
from capacity_radar.api.routes import health, insights
from capacity_radar.db.engine import create_engine, create_session_factory, init_db
from capacity_radar.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Code before `yield` runs once at startup; code after it runs once at shutdown.
    engine = create_engine(get_settings())
    await init_db(engine)  # create missing tables; fails fast if the database is unreachable

    # Shared with request handlers through app.state.
    app.state.engine = engine
    app.state.session_factory = create_session_factory(engine)

    try:
        yield
    finally:
        await engine.dispose()  # close pooled connections, even if startup or a request failed


def create_app() -> FastAPI:
    # A factory rather than a bare global, so tests can build a fresh app per test.
    app = FastAPI(title="Review Capacity Radar", version="0.1.0", lifespan=lifespan)
    register_error_handlers(app)
    app.include_router(health.router)
    app.include_router(insights.router)
    return app


app = create_app()
