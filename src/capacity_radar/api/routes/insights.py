"""GET /api/v1/repos/{owner}/{repo}/insights/review-capacity"""

from datetime import date, timedelta
from typing import Annotated

from fastapi import APIRouter, Path, Query
from sqlalchemy import select

from capacity_radar.analytics import queries
from capacity_radar.analytics.metrics import (
    CapacityReport,
    capacity_report,
    reviewer_concentration,
)
from capacity_radar.analytics.periods import Window, WindowError, make_window
from capacity_radar.api import schemas
from capacity_radar.api.deps import SessionDep, SettingsDep
from capacity_radar.api.errors import ProblemError
from capacity_radar.db.models import Repository, SyncState

router = APIRouter(prefix="/api/v1", tags=["insights"])

# GitHub owner and repo names: letters, digits, '.', '-', '_'. Checked before any lookup.
Name = Annotated[str, Path(pattern=r"^[A-Za-z0-9._-]{1,100}$")]

PROBLEM = {"content": {"application/problem+json": {}}}


@router.get(
    "/repos/{owner}/{repo}/insights/review-capacity",
    response_model=schemas.CapacityResponse,
    responses={400: PROBLEM, 404: PROBLEM, 422: PROBLEM},
)
async def review_capacity(
    owner: Name,
    repo: Name,
    session: SessionDep,
    settings: SettingsDep,
    first_day: Annotated[date | None, Query(alias="from")] = None,
    last_day: Annotated[date | None, Query(alias="to")] = None,
) -> schemas.CapacityResponse:
    """Review supply and demand for a repository over a date range (UTC, both ends
    inclusive). Defaults to the 12 weeks ending at the latest synced data."""
    if {owner, repo} & {".", ".."}:
        raise ProblemError(400, "'.' and '..' are not repository names")

    repository = await session.scalar(
        select(Repository).where(Repository.owner == owner.lower(), Repository.name == repo.lower())
    )
    state = await session.get(SyncState, repository.id) if repository else None
    if repository is None or state is None or state.history_start is None:
        raise ProblemError(404, f"{owner}/{repo} has not been synced yet; run a sync first")
    data_as_of = state.last_success_at or state.history_start

    try:
        window = make_window(
            first_day,
            last_day,
            history_start=state.history_start,
            data_as_of=data_as_of,
            max_days=settings.max_window_days,
        )
    except WindowError as exc:
        raise ProblemError(400, str(exc)) from exc

    facts = await queries.load_pr_facts(session, repository.id, window)
    pairs = await queries.load_review_pairs(session, repository.id, window)
    quality = await queries.load_data_quality(session, repository.id, window)
    report = capacity_report(facts, window)

    return schemas.CapacityResponse(
        repo=f"{repository.owner}/{repository.name}",
        window=schemas.WindowOut(
            start=window.start.date(), end=_last_day(window), days=int(window.days)
        ),
        data_as_of=data_as_of,
        summary=_summary(report),
        flow_balance=schemas.FlowBalanceOut(
            queue_at_start=report.flow_balance.queue_at_start,
            arrivals=report.flow_balance.arrivals,
            served=report.flow_balance.served,
            exited_unreviewed=report.flow_balance.exited_unreviewed,
            queue_at_end=report.flow_balance.queue_at_end,
            balanced=report.flow_balance.balanced,
        ),
        littles_law=schemas.LittlesLawOut(
            arrival_rate_per_day=_round(report.littles_law.arrival_rate_per_day, 2),
            mean_days_in_queue=_round(report.littles_law.mean_days_in_queue, 1),
            predicted_queue=_round(report.littles_law.predicted_queue, 1),
            measured_queue=_round(report.littles_law.measured_queue, 1),
            gap_percent=_round(report.littles_law.gap_percent, 0),
        ),
        reviewers=schemas.Reviewers(**_rounded_shares(reviewer_concentration(pairs))),
        weeks=[
            schemas.Week(
                start=week.start.date(),
                end=(week.end - timedelta(days=1)).date(),
                partial=week.partial,
                arrivals=week.arrivals,
                served=week.served,
                exited_unreviewed=week.exited_unreviewed,
                queue_at_end=week.queue_at_end,
                capacity_limited=week.capacity_limited,
                median_wait_hours=_round(week.median_wait_hours, 1),
            )
            for week in report.weeks
        ],
        data_quality=schemas.DataQualityOut(
            history_start=state.history_start,
            prs_created=quality.prs_created,
            bot_pr_share=_share(quality.bot_prs, quality.prs_created),
            drafts_excluded=quality.drafts_excluded,
            merged_with_review_share=_share(quality.merged_with_review, quality.merged_prs),
            capacity_limited_weeks=sum(week.capacity_limited for week in report.weeks),
        ),
    )


def _summary(report: CapacityReport) -> schemas.Summary:
    return schemas.Summary(
        arrivals=report.arrivals,
        served=report.served,
        exited_unreviewed=schemas.ExitedUnreviewed(
            merged=report.exited_unreviewed_merged, closed=report.exited_unreviewed_closed
        ),
        wait_hours=schemas.WaitHours(
            median=_round(report.wait_hours_median, 1),
            p90=_round(report.wait_hours_p90, 1),
            sample_size=report.wait_sample_size,
        ),
        queue=schemas.Queue(
            at_start=report.queue_at_start,
            at_end=report.queue_at_end,
            at_end_fresh=report.queue_at_end - report.queue_at_end_stale,
            at_end_stale=report.queue_at_end_stale,
            still_waiting_median_age_hours=_round(report.still_waiting_median_age_hours, 1),
        ),
    )


def _last_day(window: Window) -> date:
    return (window.end - timedelta(days=1)).date()  # the window's end is exclusive


def _rounded_shares(concentration: dict) -> dict:
    return {
        **concentration,
        "top1_share": _round(concentration["top1_share"], 3),
        "top3_share": _round(concentration["top3_share"], 3),
    }


def _share(part: int, whole: int) -> float | None:
    return round(part / whole, 3) if whole else None


def _round(value: float | None, digits: int) -> float | None:
    return round(value, digits) if value is not None else None
