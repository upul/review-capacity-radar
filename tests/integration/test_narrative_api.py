"""GET .../insights/narrative end to end: Postgres, the facts, a fake LLM and the cache."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import httpx
import pytest

from capacity_radar.api.deps import get_llm, get_session
from capacity_radar.db.models import PullRequest, Repository, Review, SyncState
from capacity_radar.main import create_app
from capacity_radar.settings import Settings, get_settings

URL = "/api/v1/repos/acme/widgets/insights/narrative?from=2026-09-01&to=2026-09-28"
GROUNDED = {
    "summary": "3 PRs arrived and 1 got a first review [F1].",
    "hypothesis": "There is too little activity to point to a cause.",
    "evidence": ["F1", "F2"],
    "confidence": 0.6,
}
INVENTED = {**GROUNDED, "summary": "30 PRs arrived [F1]."}


def at(month: int, day: int) -> datetime:
    return datetime(2026, month, day, 12, tzinfo=UTC)


class FakeLLM:
    model = "fake-model"

    def __init__(self, answer: dict) -> None:
        self.answer, self.calls = answer, 0

    async def draft(self, facts, feedback=None):
        self.calls += 1
        return self.answer


@pytest.fixture
async def seeded(session_factory):
    """Three PRs in September: one reviewed, one closed unreviewed, one still waiting."""
    async with session_factory() as session, session.begin():
        session.add(Repository(id=1, owner="acme", name="widgets"))
        session.add(SyncState(repo_id=1, history_start=at(8, 1), last_success_at=at(10, 20)))
        await session.flush()
        for number, created, closed in [
            (1, at(9, 2), None),
            (2, at(9, 4), at(9, 5)),
            (3, at(9, 8), None),
        ]:
            session.add(
                PullRequest(
                    id=number,
                    repo_id=1,
                    number=number,
                    author_login="alice",
                    author_is_bot=False,
                    is_draft=False,
                    created_at=created,
                    merged_at=None,
                    closed_at=closed,
                    updated_at=created,
                )
            )
        await session.flush()
        session.add(
            Review(
                id=10,
                pr_id=1,
                reviewer_login="bob",
                reviewer_is_bot=False,
                state="APPROVED",
                submitted_at=at(9, 3),
            )
        )
    return session_factory


def client_for(session_factory, llm: FakeLLM) -> httpx.AsyncClient:
    async def session() -> AsyncIterator:
        async with session_factory() as s:
            yield s

    app = create_app()
    app.dependency_overrides[get_session] = session
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_settings] = lambda: Settings(
        database_url="postgresql+asyncpg://unused",
        github_token="unused",  # noqa: S106 - a placeholder, not a secret
        llm_api_key="unused",
        llm_model=llm.model,
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test")


async def test_a_grounded_answer_is_returned_then_cached(seeded):
    llm = FakeLLM(GROUNDED)
    async with client_for(seeded, llm) as client:
        first = (await client.get(URL)).json()
        second = (await client.get(URL)).json()

    assert first["source"] == "llm"
    assert first["narrative"]["summary"] == "3 PRs arrived and 1 got a first review [F1]."
    assert first["narrative"]["confidence"] == 0.5  # capped: 1 reviewed PR, 4 full weeks
    assert [fact["id"] for fact in first["evidence"]] == ["F1", "F2"]
    assert first["evidence"][0]["text"].startswith("From 2026-09-01 to 2026-09-28, 3 PRs arrived")

    assert second["source"] == "cache" and second["narrative"] == first["narrative"]
    assert llm.calls == 1  # the second request never reached the LLM


async def test_an_invented_number_falls_back_and_is_not_cached(seeded):
    llm = FakeLLM(INVENTED)
    async with client_for(seeded, llm) as client:
        first = (await client.get(URL)).json()
        await client.get(URL)

    assert (first["source"], first["narrative"]["confidence"]) == ("fallback", 0.3)
    assert first["problems"] == ["summary: 30 is not in F1"]
    assert llm.calls == 4  # two attempts per request: fallbacks are not cached


async def test_an_unsynced_repo_is_not_found(seeded):
    async with client_for(seeded, FakeLLM(GROUNDED)) as client:
        response = await client.get("/api/v1/repos/acme/other/insights/narrative")
    assert response.status_code == 404
    assert response.headers["content-type"] == "application/problem+json"
