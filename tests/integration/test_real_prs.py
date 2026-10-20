"""Two real scikit-learn PRs, run through the real sync and metrics code and checked
against their GitHub pages. The JSON files in tests/fixtures/github are GitHub's own
responses for these PRs, saved unmodified. All times are UTC.

#34918 https://github.com/scikit-learn/scikit-learn/pull/34918
    2026-09-10 09:36:26  opened by AnneBeyer
               09:49:02  AnneBeyer COMMENTED       the author replying: doesn't count
               13:22:29  ogrisel APPROVED          first review: waited 3 h 46 min 3 s
               13:22:37  merged

#34893 https://github.com/scikit-learn/scikit-learn/pull/34893
    2026-09-06 18:13:50  opened by itz-puneet
               18:17:02  copilot-pull-request-reviewer[bot] COMMENTED    a bot: doesn't count
               18:48:40  itz-puneet COMMENTED      the author: doesn't count
    2026-09-07 10:20:40  closed, not merged: left the queue without a review
"""

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import respx
from pydantic import SecretStr

from capacity_radar.analytics.metrics import PrFacts, capacity_report
from capacity_radar.analytics.periods import Window
from capacity_radar.analytics.queries import load_data_quality, load_pr_facts, load_review_pairs
from capacity_radar.github.client import GITHUB_API_URL, GitHubClient
from capacity_radar.sync.service import sync_repo

FIXTURES = Path(__file__).parent.parent / "fixtures" / "github"
REPO_URL = f"{GITHUB_API_URL}/repos/scikit-learn/scikit-learn"
REPO_ID = 843222  # from repo.json
SEPTEMBER = Window(datetime(2026, 9, 1, tzinfo=UTC), datetime(2026, 10, 1, tzinfo=UTC))
LOOKBACK_WEEKS = 52 * 100  # far enough back that these PRs are always included


def load(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


def utc(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


@pytest.fixture
def github() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=True) as mock:
        mock.get(REPO_URL).respond(json=load("repo.json"))
        # The first sync also lists open PRs; both of these are closed.
        mock.get(f"{REPO_URL}/pulls", params={"state": "open"}).respond(json=[])
        # The list endpoint returns the same PR objects as /pulls/{n}, newest-updated first.
        mock.get(f"{REPO_URL}/pulls", params={"state": "all"}).respond(
            json=[load("pr_34918.json"), load("pr_34893.json")]
        )
        for number in (34918, 34893):
            mock.get(f"{REPO_URL}/pulls/{number}/reviews").respond(
                json=load(f"reviews_{number}.json")
            )
        yield mock


async def test_real_prs_match_their_github_pages(session_factory, github):
    async with GitHubClient(SecretStr("not-a-real-token")) as gh:
        result = await sync_repo(
            gh, session_factory, "scikit-learn", "scikit-learn", LOOKBACK_WEEKS
        )
    assert (result.prs_synced, result.reviews_synced) == (2, 4)  # every review is stored...

    async with session_factory() as session:
        facts = {pr.number: pr for pr in await load_pr_facts(session, REPO_ID, SEPTEMBER)}
        review_pairs = await load_review_pairs(session, REPO_ID, SEPTEMBER)
        quality = await load_data_quality(session, REPO_ID, SEPTEMBER)

    # ...but only ogrisel's counts as a review: the bot and both authors are left out.
    assert review_pairs == [("ogrisel", 34918)]
    assert facts[34918] == PrFacts(
        number=34918,
        created_at=utc("2026-09-10T09:36:26"),
        first_review_at=utc("2026-09-10T13:22:29"),  # not the author's reply at 09:49:02
        merged_at=utc("2026-09-10T13:22:37"),
        closed_at=utc("2026-09-10T13:22:37"),
    )
    assert facts[34893] == PrFacts(
        number=34893,
        created_at=utc("2026-09-06T18:13:50"),
        first_review_at=None,  # the bot and the author commented; nobody reviewed
        merged_at=None,
        closed_at=utc("2026-09-07T10:20:40"),
    )

    report = capacity_report(list(facts.values()), SEPTEMBER)
    assert (report.arrivals, report.served) == (2, 1)
    assert (report.exited_unreviewed_merged, report.exited_unreviewed_closed) == (0, 1)
    assert report.wait_hours_median == pytest.approx(
        timedelta(hours=3, minutes=46, seconds=3) / timedelta(hours=1)
    )
    assert (report.queue_at_start, report.queue_at_end) == (0, 0)
    assert report.flow_balance.balanced

    assert (quality.prs_created, quality.merged_prs, quality.merged_with_review) == (2, 1, 1)
