"""Review-capacity metrics: plain functions over per-PR facts, so the maths is unit-testable.

Definitions (see NOTES.md):
- Eligible PR: author is not a bot and the PR is not currently a draft.
- Qualifying review: submitted, not by the PR's author, not by a bot.
- A PR is in the review queue from creation until its first qualifying review, or until it
  is merged or closed without one ("exited unreviewed"), whichever comes first.
"""

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import mean

import numpy as np

from capacity_radar.analytics.periods import Window

HOUR = timedelta(hours=1)
DAY = timedelta(days=1)
STALE_AFTER = timedelta(days=30)  # waiting longer than this counts as stale backlog


@dataclass(frozen=True)
class PrFacts:
    """Everything the metrics need about one eligible PR."""

    number: int
    created_at: datetime
    first_review_at: datetime | None
    merged_at: datetime | None
    closed_at: datetime | None

    @property
    def reviewed(self) -> bool:
        """Left the queue through a review (not by being closed first)."""
        return self.first_review_at is not None and (
            self.closed_at is None or self.first_review_at <= self.closed_at
        )

    @property
    def left_queue_at(self) -> datetime | None:
        if self.reviewed:
            return self.first_review_at
        return self.closed_at  # None while still waiting

    def in_queue_at(self, moment: datetime) -> bool:
        """Waiting at `moment`: arrived before it, and not yet left. Events exactly at
        `moment` belong to the period that starts there, which keeps the flow balance exact
        (a PR created at the window's start is an arrival, not part of the starting queue)."""
        left = self.left_queue_at
        return self.created_at < moment and (left is None or left >= moment)


@dataclass(frozen=True)
class WeekMetrics:
    start: datetime
    end: datetime
    partial: bool
    arrivals: int
    served: int
    exited_unreviewed: int
    queue_at_end: int
    capacity_limited: bool  # queue never ran dry, so "served" estimates capacity
    median_wait_hours: float | None


@dataclass(frozen=True)
class LittlesLaw:
    arrival_rate_per_day: float
    mean_days_in_queue: float | None
    predicted_queue: float | None  # arrival rate × mean time in queue
    measured_queue: float  # average of daily queue counts
    gap_percent: float | None  # measured vs predicted; large gaps mean the queue isn't steady


@dataclass(frozen=True)
class FlowBalance:
    """queue at start + arrivals - served - exited unreviewed = queue at end, always.
    Each term is computed separately, so a mismatch can only mean a bug."""

    queue_at_start: int
    arrivals: int
    served: int
    exited_unreviewed: int
    queue_at_end: int

    @property
    def balanced(self) -> bool:
        return (
            self.queue_at_start + self.arrivals - self.served - self.exited_unreviewed
            == self.queue_at_end
        )


@dataclass(frozen=True)
class CapacityReport:
    arrivals: int
    served: int
    exited_unreviewed_merged: int
    exited_unreviewed_closed: int
    wait_hours_median: float | None
    wait_hours_p90: float | None
    wait_sample_size: int
    queue_at_start: int
    queue_at_end: int
    queue_at_end_stale: int  # of those, waiting longer than STALE_AFTER
    still_waiting_median_age_hours: float | None
    littles_law: LittlesLaw
    flow_balance: FlowBalance
    weeks: list[WeekMetrics] = field(default_factory=list)


def capacity_report(prs: list[PrFacts], window: Window) -> CapacityReport:
    def between(moment: datetime | None, start: datetime, end: datetime) -> bool:
        return moment is not None and start <= moment < end

    weeks = []
    for bucket in window.buckets():
        served_here = [
            pr
            for pr in prs
            if pr.reviewed and between(pr.first_review_at, bucket.start, bucket.end)
        ]
        queue_start = sum(pr.in_queue_at(bucket.start) for pr in prs)
        queue_end = sum(pr.in_queue_at(bucket.end) for pr in prs)
        weeks.append(
            WeekMetrics(
                start=bucket.start,
                end=bucket.end,
                partial=bucket.partial,
                arrivals=sum(between(pr.created_at, bucket.start, bucket.end) for pr in prs),
                served=len(served_here),
                exited_unreviewed=sum(
                    not pr.reviewed and between(pr.closed_at, bucket.start, bucket.end)
                    for pr in prs
                ),
                queue_at_end=queue_end,
                capacity_limited=queue_start > 0 and queue_end > 0,
                median_wait_hours=_median([_wait_hours(pr) for pr in served_here]),
            )
        )

    served = [
        pr for pr in prs if pr.reviewed and between(pr.first_review_at, window.start, window.end)
    ]
    unreviewed_exits = [
        pr for pr in prs if not pr.reviewed and between(pr.closed_at, window.start, window.end)
    ]
    waits = [_wait_hours(pr) for pr in served]
    still_waiting = [pr for pr in prs if pr.in_queue_at(window.end)]
    arrivals = sum(between(pr.created_at, window.start, window.end) for pr in prs)
    queue_at_start = sum(pr.in_queue_at(window.start) for pr in prs)

    return CapacityReport(
        arrivals=arrivals,
        served=len(served),
        exited_unreviewed_merged=sum(pr.merged_at is not None for pr in unreviewed_exits),
        exited_unreviewed_closed=sum(pr.merged_at is None for pr in unreviewed_exits),
        wait_hours_median=_median(waits),
        wait_hours_p90=_percentile(waits, 90),
        wait_sample_size=len(waits),
        queue_at_start=queue_at_start,
        queue_at_end=len(still_waiting),
        queue_at_end_stale=sum(window.end - pr.created_at > STALE_AFTER for pr in still_waiting),
        still_waiting_median_age_hours=_median(
            [(window.end - pr.created_at) / HOUR for pr in still_waiting]
        ),
        littles_law=_littles_law(prs, window, arrivals),
        flow_balance=FlowBalance(
            queue_at_start=queue_at_start,
            arrivals=arrivals,
            served=len(served),
            exited_unreviewed=len(unreviewed_exits),
            queue_at_end=len(still_waiting),
        ),
        weeks=weeks,
    )


def _littles_law(prs: list[PrFacts], window: Window, arrivals: int) -> LittlesLaw:
    """L = λ × W, with L, λ and W measured independently, as a self-check."""
    rate = arrivals / window.days  # λ
    left_in_window = [
        pr
        for pr in prs
        if pr.left_queue_at is not None and window.start <= pr.left_queue_at < window.end
    ]
    mean_days = (  # W
        mean((pr.left_queue_at - pr.created_at) / DAY for pr in left_in_window)  # type: ignore[operator]
        if left_in_window
        else None
    )
    day_ends = [window.start + DAY * i for i in range(1, int(window.days) + 1)]
    measured = mean(sum(pr.in_queue_at(t) for pr in prs) for t in day_ends) if day_ends else 0.0
    predicted = rate * mean_days if mean_days is not None else None
    gap = (measured - predicted) / predicted * 100 if predicted else None
    return LittlesLaw(rate, mean_days, predicted, measured, gap)


def reviewer_concentration(review_pairs: list[tuple[str, int]], top: int = 5) -> dict:
    """Share of review work by the busiest reviewers, counted as distinct PRs reviewed
    (one reviewer replying 50 times on one PR counts once)."""
    per_reviewer = Counter(login for login, _ in set(review_pairs))
    total = sum(per_reviewer.values())
    ranked = per_reviewer.most_common()

    def share(n: int) -> float | None:
        return sum(count for _, count in ranked[:n]) / total if total else None

    return {
        "reviewers": len(per_reviewer),
        "top1_share": share(1),
        "top3_share": share(3),
        "top": [{"login": login, "prs_reviewed": count} for login, count in ranked[:top]],
    }


def _wait_hours(pr: PrFacts) -> float:
    return (pr.first_review_at - pr.created_at) / HOUR  # type: ignore[operator]


def _median(values: list[float]) -> float | None:
    return _percentile(values, 50)


def _percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None
