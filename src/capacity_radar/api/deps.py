from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from capacity_radar.narrative.llm import NarrativeLLM
from capacity_radar.settings import Settings, get_settings


async def get_session(request: Request) -> AsyncGenerator[AsyncSession, None]:
    async with request.app.state.session_factory() as session:
        yield session


def get_llm(request: Request) -> NarrativeLLM:
    return request.app.state.narrative_llm


SessionDep = Annotated[AsyncSession, Depends(get_session)]
LLMDep = Annotated[NarrativeLLM, Depends(get_llm)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
