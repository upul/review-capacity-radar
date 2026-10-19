"""Health check: is the service up, and can it reach its database?"""

import asyncio
import logging
from typing import Literal

from fastapi import APIRouter, Response, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from capacity_radar.api.deps import SessionDep

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])

# A health check must answer quickly, even when the database hangs instead of refusing.
# The first check on an already-open connection can take about twice this, because the
# driver also tries to cancel the stuck query on the server.
DB_CHECK_TIMEOUT_SECONDS = 2


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: Literal["ok", "unavailable"]


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": HealthResponse}},
)
async def health(session: SessionDep, response: Response) -> HealthResponse:
    try:
        async with asyncio.timeout(DB_CHECK_TIMEOUT_SECONDS):
            await session.execute(text("SELECT 1"))
    # OSError covers refused connections; TimeoutError (a subclass) covers the timeout.
    except (SQLAlchemyError, OSError):
        # Details go to the log only: error text can reveal hosts and internal setup.
        logger.warning("database health check failed", exc_info=True)
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(status="degraded", database="unavailable")
    return HealthResponse(status="ok", database="ok")
