"""The numbered facts the narrative may cite.

Every number the LLM is allowed to use is computed here, by code, and given an ID (F1, F2,
...). The LLM explains the facts and cites their IDs; it never computes anything itself.
Each fact keeps the numbers exactly as they appear in its text, so the narrative validator
can check that every number the LLM writes comes from a fact. Dates are labels, not numbers.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta

from capacity_radar.analytics.anomalies import Anomaly
from capacity_radar.analytics.forecast import ForecastWeek
from capacity_radar.analytics.metrics import STALE_AFTER, CapacityReport
from capacity_radar.analytics.periods import Window
from capacity_radar.analytics.queries import DataQuality

ANOMALY_LABELS = {  # metric -> (how it reads in a sentence, unit)
    "arrivals": ("arrivals", ""),
    "served": ("first reviews", ""),
    "median_wait_hours": ("median wait for a first review", " h"),
}


@dataclass(frozen=True)
class Fact:
    id: str
    text: str
    numbers: tuple[float, ...]  # the numbers in `text`, as written there


@dataclass
class _FactList:
    facts: list[Fact] = field(default_factory=list)

    def add(self, template: str, *numbers: float | None, **labels: str) -> None:
        """Add a fact unless one of its numbers is missing. `{}` in the template takes the
        numbers in order; `{name}` takes a label such as a date."""
        if any(number is None for number in numbers):
            return
        shown = tuple(_rounded(number) for number in numbers)  # type: ignore[arg-type]
        text = template.format(*map(str, shown), **labels)
        self.facts.append(Fact(f"F{len(self.facts) + 1}", text, shown))


def build_facts(
    window: Window,
    report: CapacityReport,
    reviewers: dict,
    quality: DataQuality,
    anomalies: Sequence[Anomaly],
    forecast: Sequence[ForecastWeek] = (),
) -> list[Fact]:
    facts = _FactList()
    last_day = (window.end - timedelta(days=1)).date()
    facts.add(
        "From {start} to {end}, {} PRs arrived; {} got a first review and {} left without one "
        "({} merged, {} closed)",
        report.arrivals,
        report.served,
        report.exited_unreviewed_merged + report.exited_unreviewed_closed,
        report.exited_unreviewed_merged,
        report.exited_unreviewed_closed,
        start=str(window.start.date()),
        end=str(last_day),
    )
    facts.add(
        "PRs waiting for a first review: {} at the start of the window, {} at the end",
        report.queue_at_start,
        report.queue_at_end,
    )
    if report.queue_at_end:
        facts.add(
            "{} of the {} waiting PRs ({}%) had waited more than {} days; "
            "the median waiting PR was {} days old",
            report.queue_at_end_stale,
            report.queue_at_end,
            _percent(report.queue_at_end_stale, report.queue_at_end),
            STALE_AFTER.days,
            _days(report.still_waiting_median_age_hours),
        )
    facts.add(
        "Time to first review: median {} h, p90 {} h, across {} reviewed PRs",
        report.wait_hours_median,
        report.wait_hours_p90,
        report.wait_sample_size,
    )
    full_weeks = [week for week in report.weeks if not week.partial]
    facts.add(
        "{} of {} full weeks were capacity-limited: the queue never emptied",
        sum(week.capacity_limited for week in full_weeks),
        len(full_weeks),
    )
    facts.add(
        "{} people reviewed; the busiest reviewed {}% of the reviewed PRs, the top three {}%",
        reviewers["reviewers"],
        _percent(reviewers["top1_share"]),
        _percent(reviewers["top3_share"]),
    )
    facts.add(
        "{}% of merged PRs had a GitHub review",
        _percent(quality.merged_with_review, quality.merged_prs),
    )
    littles_law = report.littles_law
    facts.add(
        "Little's Law check: arrival rate x mean time in queue predicts an average queue of {}; "
        "the measured average was {} ({}% apart)",
        littles_law.predicted_queue,
        littles_law.measured_queue,
        littles_law.gap_percent,
    )
    for anomaly in anomalies:
        label, unit = ANOMALY_LABELS[anomaly.metric]
        facts.add(
            f"Week of {{week}}: {label} {{}}{unit} vs {{}}{unit} on average in the other weeks "
            "(z = {})",
            anomaly.value,
            anomaly.others_mean,
            anomaly.z,
            week=str(anomaly.week_start.date()),
        )
    if forecast:
        horizon = forecast[-1]
        facts.add(
            "Forecast {} weeks ahead: queue p50 {}, p10 {}, p90 {}; "
            "{}% chance it ends above today's queue",
            horizon.week,
            horizon.p50,
            horizon.p10,
            horizon.p90,
            _percent(horizon.chance_queue_grows),
        )
    return facts.facts


def _percent(part: float | None, whole: float = 1) -> float | None:
    return round(part / whole * 100) if part is not None and whole else None


def _days(hours: float | None) -> float | None:
    return hours / 24 if hours is not None else None


def _rounded(number: float) -> float:
    rounded = round(number, 1)
    return int(rounded) if rounded == int(rounded) else rounded
