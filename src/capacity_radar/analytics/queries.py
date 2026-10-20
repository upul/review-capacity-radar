"""Database reads for the capacity metrics. Only facts are fetched here; all the maths
lives in metrics.py, where it is unit-tested without a database."""

from dataclasses import dataclass

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from capacity_radar.analytics.metrics import PrFacts
from capacity_radar.analytics.periods import Window
from capacity_radar.db.models import PullRequest, Review

# The PR is one an author is asking a human to review: not opened by a bot, not a draft.
ELIGIBLE = and_(PullRequest.author_is_bot.is_(False), PullRequest.is_draft.is_(False))

# A review that counts: actually submitted, by someone other than the author, not a bot.
# Excluding the author matters: their replies in review threads are stored as COMMENTED reviews.
QUALIFYING_REVIEW = and_(
    Review.submitted_at.is_not(None),
    Review.reviewer_is_bot.is_(False),
    Review.reviewer_login != PullRequest.author_login,
)


@dataclass(frozen=True)
class DataQuality:
    prs_created: int  # every PR created in the window, bots and drafts included
    bot_prs: int
    drafts_excluded: int
    merged_prs: int  # eligible PRs merged in the window
    merged_with_review: int  # ...of which had at least one qualifying review


async def load_pr_facts(session: AsyncSession, repo_id: int, window: Window) -> list[PrFacts]:
    """One row per eligible PR that could be in the queue during the window: created before
    it ends, and not closed before it starts."""
    first_review = (
        select(func.min(Review.submitted_at))
        .where(Review.pr_id == PullRequest.id, QUALIFYING_REVIEW)
        .correlate(PullRequest)
        .scalar_subquery()
    )
    rows = await session.execute(
        select(
            PullRequest.number,
            PullRequest.created_at,
            first_review,
            PullRequest.merged_at,
            PullRequest.closed_at,
        ).where(
            PullRequest.repo_id == repo_id,
            ELIGIBLE,
            PullRequest.created_at < window.end,
            or_(PullRequest.closed_at.is_(None), PullRequest.closed_at >= window.start),
        )
    )
    return [PrFacts(*row) for row in rows]


async def load_review_pairs(
    session: AsyncSession, repo_id: int, window: Window
) -> list[tuple[str, int]]:
    """(reviewer, PR number) for every qualifying review submitted in the window."""
    rows = await session.execute(
        select(Review.reviewer_login, PullRequest.number)
        .join(PullRequest, Review.pr_id == PullRequest.id)
        .where(
            PullRequest.repo_id == repo_id,
            ELIGIBLE,
            QUALIFYING_REVIEW,
            Review.submitted_at >= window.start,
            Review.submitted_at < window.end,
        )
    )
    return [(reviewer, number) for reviewer, number in rows]


async def load_data_quality(session: AsyncSession, repo_id: int, window: Window) -> DataQuality:
    in_window = and_(
        PullRequest.repo_id == repo_id,
        PullRequest.created_at >= window.start,
        PullRequest.created_at < window.end,
    )
    created = await session.execute(
        select(
            func.count(),
            func.count().filter(PullRequest.author_is_bot),
            func.count().filter(PullRequest.is_draft, PullRequest.author_is_bot.is_(False)),
        ).where(in_window)
    )
    prs_created, bot_prs, drafts = created.one()

    has_review = select(Review.id).where(Review.pr_id == PullRequest.id, QUALIFYING_REVIEW).exists()
    merged = await session.execute(
        select(func.count(), func.count().filter(has_review)).where(
            PullRequest.repo_id == repo_id,
            ELIGIBLE,
            PullRequest.merged_at >= window.start,
            PullRequest.merged_at < window.end,
        )
    )
    merged_prs, merged_with_review = merged.one()
    return DataQuality(prs_created, bot_prs, drafts, merged_prs, merged_with_review)
