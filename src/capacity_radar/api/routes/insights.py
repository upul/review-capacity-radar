"""GET /api/v1/repos/{owner}/{repo}/insights/review-capacity
GET /api/v1/repos/{owner}/{repo}/insights/queue-forecast
GET /api/v1/repos/{owner}/{repo}/insights/narrative"""

from datetime import date, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Path, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from capacity_radar.analytics import queries
from capacity_radar.analytics.anomalies import find_anomalies
from capacity_radar.analytics.forecast import MIN_HISTORY_WEEKS, RUNS, SEED, forecast_queue
from capacity_radar.analytics.metrics import (
    CapacityReport,
    capacity_report,
    reviewer_concentration,
)
from capacity_radar.analytics.periods import (
    DEFAULT_WEEKS,
    Window,
    WindowError,
    make_window,
    recent_full_weeks,
)
from capacity_radar.api import schemas
from capacity_radar.api.deps import LLMDep, SessionDep, SettingsDep
from capacity_radar.api.errors import ProblemError
from capacity_radar.db.models import Repository, SyncState
from capacity_radar.narrative.evidence import Fact, build_facts
from capacity_radar.narrative.llm import PROMPT_VERSION
from capacity_radar.narrative.service import cache_key, cached, store, write_narrative
from capacity_radar.settings import Settings

router = APIRouter(prefix="/api/v1", tags=["insights"])

# GitHub owner and repo names: letters, digits, '.', '-', '_'. Checked before any lookup.
Name = Annotated[str, Path(pattern=r"^[A-Za-z0-9._-]{1,100}$")]

PROBLEM = {"content": {"application/problem+json": {}}}

NARRATIVE_WEEKS = 4  # how far ahead the narrative's forecast fact looks


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
    inclusive). Defaults to the 12 weeks ending at the last complete day of data."""
    repository, history_start, data_as_of = await _synced_repo(session, owner, repo)
    window = _window(first_day, last_day, history_start, data_as_of, settings)

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
            history_start=history_start,
            prs_created=quality.prs_created,
            bot_pr_share=_share(quality.bot_prs, quality.prs_created),
            drafts_excluded=quality.drafts_excluded,
            merged_with_review_share=_share(quality.merged_with_review, quality.merged_prs),
            capacity_limited_weeks=sum(week.capacity_limited for week in report.weeks),
        ),
    )


@router.get(
    "/repos/{owner}/{repo}/insights/queue-forecast",
    response_model=schemas.QueueForecastResponse,
    responses={400: PROBLEM, 404: PROBLEM, 422: PROBLEM},
)
async def queue_forecast(
    owner: Name,
    repo: Name,
    session: SessionDep,
    settings: SettingsDep,
    weeks: Annotated[int, Query(ge=1, description="weeks ahead to forecast")] = 4,
) -> schemas.QueueForecastResponse:
    """How big the review queue may get in the coming weeks, if they look like the recent
    past: 2,000 simulated futures, each replaying randomly chosen weeks from the last 12."""
    if weeks > settings.max_forecast_weeks:
        raise ProblemError(400, f"'weeks' can be at most {settings.max_forecast_weeks}")
    repository, history_start, data_as_of = await _synced_repo(session, owner, repo)

    window, history = await _recent_history(session, repository.id, history_start, data_as_of)
    if len(history.weeks) < MIN_HISTORY_WEEKS:
        raise ProblemError(
            400, f"need {MIN_HISTORY_WEEKS} complete weeks of synced data to forecast"
        )
    net_changes = _net_changes(history)
    forecast = forecast_queue(history.queue_at_end, net_changes, weeks)

    return schemas.QueueForecastResponse(
        repo=f"{repository.owner}/{repository.name}",
        data_as_of=data_as_of,
        queue_now=history.queue_at_end,
        history=schemas.ForecastHistory(
            start=window.start.date(),
            end=_last_day(window),
            weeks=[
                schemas.HistoryWeek(
                    start=week.start.date(),
                    end=(week.end - timedelta(days=1)).date(),
                    arrivals=week.arrivals,
                    served=week.served,
                    exited_unreviewed=week.exited_unreviewed,
                    net_change=net,
                )
                for week, net in zip(history.weeks, net_changes, strict=True)
            ],
        ),
        simulation=schemas.Simulation(runs=RUNS, seed=SEED),
        forecast=[
            schemas.ForecastWeekOut(
                week=week.week,
                ends_on=_last_day(window) + timedelta(weeks=week.week),
                p10=round(week.p10, 1),
                p50=round(week.p50, 1),
                p90=round(week.p90, 1),
                chance_queue_grows=round(week.chance_queue_grows, 3),
            )
            for week in forecast
        ],
    )


@router.get(
    "/repos/{owner}/{repo}/insights/narrative",
    response_model=schemas.NarrativeResponse,
    responses={400: PROBLEM, 404: PROBLEM, 422: PROBLEM},
)
async def narrative(
    owner: Name,
    repo: Name,
    session: SessionDep,
    settings: SettingsDep,
    llm: LLMDep,
    first_day: Annotated[date | None, Query(alias="from")] = None,
    last_day: Annotated[date | None, Query(alias="to")] = None,
) -> schemas.NarrativeResponse:
    """What happened to the review queue and the most likely cause, written by an LLM from
    numbered facts. Every number in it is checked against the facts it cites; if the LLM's
    answer fails those checks twice, a plain summary of the facts is returned instead."""
    # 1. The facts, computed by code: the same numbers as review-capacity, plus anomalies.
    repository, history_start, data_as_of = await _synced_repo(session, owner, repo)
    window = _window(first_day, last_day, history_start, data_as_of, settings)
    report = capacity_report(await queries.load_pr_facts(session, repository.id, window), window)
    reviewers = reviewer_concentration(
        await queries.load_review_pairs(session, repository.id, window)
    )
    quality = await queries.load_data_quality(session, repository.id, window)
    forecast = []
    history_window, history = await _recent_history(
        session, repository.id, history_start, data_as_of
    )
    if window.end == history_window.end and len(history.weeks) >= MIN_HISTORY_WEEKS:
        # Only a window ending today has a "today's queue" for the forecast to start from.
        forecast = forecast_queue(history.queue_at_end, _net_changes(history), NARRATIVE_WEEKS)
    facts = build_facts(window, report, reviewers, quality, find_anomalies(report.weeks), forecast)

    # 2. The narrative: from the cache if these exact facts were explained before.
    key = cache_key(repository.id, llm.model, facts)
    result = await cached(session, key)
    if result is None:
        await session.close()  # give the connection back to the pool during the slow LLM call
        result = await write_narrative(llm, facts, report, quality)
        await store(session, key, repository.id, llm.model, result)

    by_id = {fact.id: fact for fact in facts}
    return schemas.NarrativeResponse(
        repo=f"{repository.owner}/{repository.name}",
        window=schemas.WindowOut(
            start=window.start.date(), end=_last_day(window), days=int(window.days)
        ),
        data_as_of=data_as_of,
        narrative=schemas.NarrativeOut(
            summary=result.summary,
            hypothesis=result.hypothesis,
            confidence=result.confidence,
            confidence_notes=result.confidence_notes,
        ),
        evidence=[_fact_out(by_id[fact_id]) for fact_id in result.evidence],
        facts=[_fact_out(fact) for fact in facts],
        source=result.source,
        model=llm.model,
        prompt_version=PROMPT_VERSION,
        problems=result.problems,
    )


def _window(
    first_day: date | None,
    last_day: date | None,
    history_start: datetime,
    data_as_of: datetime,
    settings: Settings,
) -> Window:
    """The requested window, checked against the synced data; 400 if it doesn't fit."""
    try:
        return make_window(
            first_day,
            last_day,
            history_start=history_start,
            data_as_of=data_as_of,
            max_days=settings.max_window_days,
        )
    except WindowError as exc:
        raise ProblemError(400, str(exc)) from exc


async def _recent_history(
    session: AsyncSession, repo_id: int, history_start: datetime, data_as_of: datetime
) -> tuple[Window, CapacityReport]:
    """The last 12 complete weeks, which the forecast replays."""
    window = recent_full_weeks(
        history_start=history_start, data_as_of=data_as_of, weeks=DEFAULT_WEEKS
    )
    return window, capacity_report(await queries.load_pr_facts(session, repo_id, window), window)


def _net_changes(history: CapacityReport) -> list[int]:
    return [week.arrivals - week.served - week.exited_unreviewed for week in history.weeks]


def _fact_out(fact: Fact) -> schemas.FactOut:
    return schemas.FactOut(id=fact.id, text=fact.text)


async def _synced_repo(
    session: AsyncSession, owner: str, repo: str
) -> tuple[Repository, datetime, datetime]:
    """The repository, where its synced data starts, and how recent it is; 404 if the
    repository was never synced."""
    if {owner, repo} & {".", ".."}:
        raise ProblemError(400, "'.' and '..' are not repository names")
    repository = await session.scalar(
        select(Repository).where(Repository.owner == owner.lower(), Repository.name == repo.lower())
    )
    state = await session.get(SyncState, repository.id) if repository else None
    if repository is None or state is None or state.history_start is None:
        raise ProblemError(404, f"{owner}/{repo} has not been synced yet; run a sync first")
    return repository, state.history_start, state.last_success_at or state.history_start


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
