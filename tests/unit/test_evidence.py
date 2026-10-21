"""The facts for the hand-worked example in test_metrics.py (5 to 18 Jan 2026)."""

import re
from datetime import UTC, datetime

import pytest
from test_metrics import PRS, WINDOW  # the example worked out by hand there

from capacity_radar.analytics.anomalies import Anomaly
from capacity_radar.analytics.forecast import ForecastWeek
from capacity_radar.analytics.metrics import capacity_report, reviewer_concentration
from capacity_radar.analytics.queries import DataQuality
from capacity_radar.narrative.evidence import build_facts

REVIEWERS = reviewer_concentration([("ana", 1), ("ana", 2), ("ben", 2)])  # ana 2 PRs, ben 1
QUALITY = DataQuality(
    prs_created=5, bot_prs=0, drafts_excluded=0, merged_prs=1, merged_with_review=0
)
WAIT_SPIKE = Anomaly(
    metric="median_wait_hours",
    week_start=datetime(2026, 1, 12, tzinfo=UTC),
    value=90,
    others_mean=35.6,
    others_sd=3.847,
    z=14.14,
)
FORECAST = [
    ForecastWeek(week=1, p10=1, p50=2, p90=3, chance_queue_grows=0.5),
    ForecastWeek(week=2, p10=0, p50=2.5, p90=4, chance_queue_grows=0.613),
]


@pytest.fixture(scope="module")
def facts():
    report = capacity_report(PRS, WINDOW)
    return build_facts(WINDOW, report, REVIEWERS, QUALITY, [WAIT_SPIKE], FORECAST)


def test_the_facts(facts):
    texts = {fact.id: fact.text for fact in facts}
    assert texts == {
        "F1": "From 2026-01-05 to 2026-01-18, 5 PRs arrived; 2 got a first review and 3 left "
        "without one (1 merged, 2 closed)",
        "F2": "PRs waiting for a first review: 1 at the start of the window, 1 at the end",
        "F3": "0 of the 1 waiting PRs (0%) had waited more than 30 days; "
        "the median waiting PR was 20 days old",  # PR 4: 30 Dec to 19 Jan
        "F4": "Time to first review: median 96 h, p90 153.6 h, across 2 reviewed PRs",
        "F5": "2 of 2 full weeks were capacity-limited: the queue never emptied",
        "F6": "2 people reviewed; the busiest reviewed 67% of the reviewed PRs, the top three 100%",
        "F7": "0% of merged PRs had a GitHub review",
        "F8": texts["F8"],  # Little's Law: its maths is tested in test_metrics.py
        "F9": "Week of 2026-01-12: median wait for a first review 90 h vs 35.6 h on average in "
        "the other weeks (z = 14.1)",
        "F10": "Forecast 2 weeks ahead: queue p50 2.5, p10 0, p90 4; "
        "61% chance it ends above today's queue",
    }
    assert texts["F8"].startswith("Little's Law check:")


def test_every_number_in_the_text_is_recorded(facts):
    # The validator relies on this: the numbers of a fact are exactly those in its text.
    for fact in facts:
        text = re.sub(r"\d{4}-\d{2}-\d{2}", "", fact.text)  # dates are labels, not numbers
        text = text.replace("p10", "").replace("p50", "").replace("p90", "")  # names, too
        in_text = [float(number) for number in re.findall(r"-?\d+(?:\.\d+)?", text)]
        assert in_text == [float(number) for number in fact.numbers], fact.text


def test_facts_with_missing_numbers_are_left_out():
    nobody_reviewed = [pr for pr in PRS if pr.first_review_at is None]
    report = capacity_report(nobody_reviewed, WINDOW)
    facts = build_facts(WINDOW, report, reviewer_concentration([]), QUALITY, [])

    assert not any(fact.text.startswith(("Time to first review", "0 people")) for fact in facts)
    assert [fact.id for fact in facts] == [f"F{i}" for i in range(1, len(facts) + 1)]  # no gaps
