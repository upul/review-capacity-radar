"""Unusual weeks, found with a leave-one-out z-score.

Each week is compared with the mean and standard deviation of the *other* weeks, so an
extreme week can't inflate the standard deviation and hide itself. A plain z-score over all
weeks can't even reach 3 for one outlier among few weeks: the most it can be is
(n - 1) / sqrt(n), 3.02 for 11 weeks.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from statistics import mean, stdev

from capacity_radar.analytics.metrics import WeekMetrics

ANOMALY_Z = 3  # three-sigma rule: a normal week is this far out about 0.3% of the time
MIN_OTHER_WEEKS = 4  # fewer weeks than this to compare against is too little to judge
METRICS = ("arrivals", "served", "median_wait_hours")


@dataclass(frozen=True)
class Anomaly:
    metric: str  # one of METRICS
    week_start: datetime
    value: float
    others_mean: float
    others_sd: float
    z: float  # positive: higher than usual; negative: lower


@dataclass(frozen=True)
class Score:
    z: float
    others_mean: float
    others_sd: float


def find_anomalies(weeks: Sequence[WeekMetrics]) -> list[Anomaly]:
    """Weeks whose arrivals, first reviews or median wait stand out, most unusual first.
    The partial last week is left out: its counts are low only because it is short."""
    full_weeks = [week for week in weeks if not week.partial]
    anomalies = []
    for metric in METRICS:
        # A week where nobody was reviewed has no median wait; it is left out of that metric.
        scored = [(week, getattr(week, metric)) for week in full_weeks]
        scored = [(week, value) for week, value in scored if value is not None]
        values = [value for _, value in scored]
        for (week, value), score in zip(scored, leave_one_out_z(values), strict=True):
            if score is not None and abs(score.z) > ANOMALY_Z:
                anomalies.append(
                    Anomaly(metric, week.start, value, score.others_mean, score.others_sd, score.z)
                )
    return sorted(anomalies, key=lambda anomaly: abs(anomaly.z), reverse=True)


def leave_one_out_z(values: Sequence[float]) -> list[Score | None]:
    """Each value's z-score against the mean and sample SD of all the other values.
    None when it can't be scored: too few other values, or they are all identical."""
    scores: list[Score | None] = []
    for i, value in enumerate(values):
        others = [*values[:i], *values[i + 1 :]]
        if len(others) < MIN_OTHER_WEEKS:
            scores.append(None)
            continue
        others_mean, others_sd = mean(others), stdev(others)  # stdev divides by n - 1
        if others_sd == 0:
            scores.append(None)
            continue
        scores.append(Score((value - others_mean) / others_sd, others_mean, others_sd))
    return scores
