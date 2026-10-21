"""Caps on the LLM's confidence: it can't be more sure than the data allows.

The LLM proposes a confidence from how well the facts fit its hypothesis. Code then caps it
by how much data there is, which the LLM can't judge reliably.
"""

from dataclasses import dataclass

from capacity_radar.analytics.metrics import CapacityReport
from capacity_radar.analytics.queries import DataQuality

MIN_REVIEWED_PRS = 30
MIN_FULL_WEEKS = 8
MIN_MERGED_REVIEW_SHARE = 0.5


@dataclass(frozen=True)
class Confidence:
    value: float
    notes: list[str]  # why it was lowered; empty if the LLM's value stood


def capped_confidence(proposed: float, report: CapacityReport, quality: DataQuality) -> Confidence:
    full_weeks = sum(not week.partial for week in report.weeks)
    caps = []
    if report.wait_sample_size < MIN_REVIEWED_PRS:
        caps.append((0.5, f"only {report.wait_sample_size} PRs got a first review in the window"))
    if full_weeks < MIN_FULL_WEEKS:
        caps.append((0.6, f"only {full_weeks} full weeks of data"))
    if (
        quality.merged_prs
        and quality.merged_with_review / quality.merged_prs < MIN_MERGED_REVIEW_SHARE
    ):
        caps.append(
            (0.5, "under half of merged PRs had a GitHub review: review may happen elsewhere")
        )

    applied = [(cap, note) for cap, note in caps if cap < proposed]
    value = min([proposed, *(cap for cap, _ in applied)])
    return Confidence(round(value, 2), [f"capped at {cap}: {note}" for cap, note in applied])
