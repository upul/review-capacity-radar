"""Incremental sync of one repository's pull requests and reviews into Postgres.

Each run fetches only PRs whose `updated_at` moved since the previous run (the cursor).
That relies on GitHub updating a PR's `updated_at` whenever it gets a review, which we
verified on 50 real scikit-learn PRs. Writes are upserts keyed on GitHub's IDs, so any run
can safely be repeated; the cursor only advances after every PR in the run was saved.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from capacity_radar.db.models import PullRequest as PullRequestRow
from capacity_radar.db.models import Repository, SyncState
from capacity_radar.db.models import Review as ReviewRow
from capacity_radar.github.client import GitHubClient, GitHubRepository, PullRequest, Review

logger = logging.getLogger(__name__)

PROGRESS_EVERY = 100  # log a progress line every N pull requests saved


@dataclass(frozen=True)
class SyncResult:
    repo: str
    first_run: bool
    prs_synced: int
    prs_unchanged: int  # listed, but already stored with the same updated_at
    reviews_synced: int
    requests_made: int
    cursor: datetime | None


async def sync_repo(
    gh: GitHubClient,
    session_factory: async_sessionmaker[AsyncSession],
    owner: str,
    repo: str,
    lookback_weeks: int,
) -> SyncResult:
    requests_before = gh.requests_made
    info = await gh.get_repository(owner, repo)
    slug = f"{info.owner.login}/{info.name}".lower()

    async with session_factory() as session, session.begin():
        await _upsert_repository(session, info)
        state = await session.get(SyncState, info.id)
        cursor = state.cursor_updated_at if state else None

    first_run = cursor is None
    stop_at = cursor or (datetime.now(UTC) - timedelta(weeks=lookback_weeks))

    # PRs come newest-updated first, so everything after the first one older than stop_at
    # is older too. A PR updated while we page moves to the top and is skipped this run,
    # but its new updated_at is newer than the cursor, so the next run picks it up.
    changed: dict[int, PullRequest] = {}
    async for pr in gh.list_pull_requests(owner, repo):
        if pr.updated_at < stop_at:
            break
        changed[pr.id] = pr

    if first_run:
        # Old open PRs nobody touched within the lookback window are still in the review
        # queue, so the first run collects every open PR as well.
        async for pr in gh.list_pull_requests(owner, repo, state="open"):
            changed.setdefault(pr.id, pr)

    # A PR already stored with the same updated_at has not changed, so neither have its
    # reviews (a new review always moves updated_at). Skipping those makes a re-run after a
    # failure cheap: only the PRs the failed run did not reach cost a reviews request.
    async with session_factory() as session:
        rows = await session.execute(
            select(PullRequestRow.id, PullRequestRow.updated_at).where(
                PullRequestRow.id.in_(changed.keys())
            )
        )
        stored = dict(rows.all())
    to_sync = [pr for pr in changed.values() if stored.get(pr.id) != pr.updated_at]

    logger.info(
        "%s: %d pull requests to sync, %d unchanged (first run: %s)",
        slug,
        len(to_sync),
        len(changed) - len(to_sync),
        first_run,
    )

    reviews_synced = 0
    for done, pr in enumerate(to_sync, start=1):
        reviews = await gh.list_reviews(owner, repo, pr.number)
        async with session_factory() as session, session.begin():  # PR and reviews together
            await _upsert_pull_request(session, info.id, pr)
            await _upsert_reviews(session, pr.id, reviews)
        reviews_synced += len(reviews)
        if done % PROGRESS_EVERY == 0:
            logger.info("%s: %d/%d pull requests saved", slug, done, len(to_sync))

    # Advance the cursor only now that every PR above was saved. If the run failed earlier,
    # the old cursor stays and the next run redoes the work; upserts make that harmless.
    new_cursor = max([pr.updated_at for pr in changed.values()], default=cursor)
    if cursor is not None and new_cursor is not None:
        new_cursor = max(new_cursor, cursor)
    async with session_factory() as session, session.begin():
        await _save_sync_state(session, info.id, new_cursor, stop_at if first_run else None)

    return SyncResult(
        repo=slug,
        first_run=first_run,
        prs_synced=len(to_sync),
        prs_unchanged=len(changed) - len(to_sync),
        reviews_synced=reviews_synced,
        requests_made=gh.requests_made - requests_before,
        cursor=new_cursor,
    )


async def _upsert_repository(session: AsyncSession, info: GitHubRepository) -> None:
    values = {"id": info.id, "owner": info.owner.login.lower(), "name": info.name.lower()}
    stmt = insert(Repository).values(values)
    await session.execute(
        stmt.on_conflict_do_update(
            index_elements=[Repository.id],
            set_={"owner": stmt.excluded.owner, "name": stmt.excluded.name},  # repo renamed
        )
    )


async def _upsert_pull_request(session: AsyncSession, repo_id: int, pr: PullRequest) -> None:
    values = {
        "id": pr.id,
        "repo_id": repo_id,
        "number": pr.number,
        "author_login": pr.author_login,
        "author_is_bot": pr.author_is_bot,
        "is_draft": pr.draft,
        "created_at": pr.created_at,
        "merged_at": pr.merged_at,
        "closed_at": pr.closed_at,
        "updated_at": pr.updated_at,
    }
    stmt = insert(PullRequestRow).values(values)
    await session.execute(
        stmt.on_conflict_do_update(
            index_elements=[PullRequestRow.id],
            set_={key: stmt.excluded[key] for key in values if key != "id"},
        )
    )


async def _upsert_reviews(session: AsyncSession, pr_id: int, reviews: list[Review]) -> None:
    if not reviews:
        return
    rows = [
        {
            "id": review.id,
            "pr_id": pr_id,
            "reviewer_login": review.reviewer_login,
            "reviewer_is_bot": review.reviewer_is_bot,
            "state": review.state,
            "submitted_at": review.submitted_at,
        }
        for review in reviews
    ]
    stmt = insert(ReviewRow).values(rows)
    await session.execute(
        stmt.on_conflict_do_update(
            index_elements=[ReviewRow.id],
            set_={key: stmt.excluded[key] for key in rows[0] if key != "id"},  # e.g. DISMISSED
        )
    )


async def _save_sync_state(
    session: AsyncSession,
    repo_id: int,
    cursor: datetime | None,
    history_start: datetime | None,
) -> None:
    """Record the new cursor. `history_start` is only passed on the first run: it marks
    where the synced data begins, so analytics can refuse windows the data doesn't cover."""
    now = datetime.now(UTC)
    values = {"cursor_updated_at": cursor, "last_success_at": now}
    if history_start is not None:
        values["history_start"] = history_start
    stmt = insert(SyncState).values(repo_id=repo_id, **values)
    await session.execute(
        stmt.on_conflict_do_update(index_elements=[SyncState.repo_id], set_=values)
    )
