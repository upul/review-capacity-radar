"""Leave-one-out z-scores on the hand-worked example: median waits of six weeks.

    Week:        W1  W2  W3  W4  W5  W6
    Median wait: 30  36  40  34  90  38

W5 against the other five (30 36 40 34 38): mean 178 / 5 = 35.6,
SD = sqrt(59.2 / 4) = 3.85, so z = (90 - 35.6) / 3.85 = 14.1: an anomaly.
W1 against 36 40 34 90 38: mean 47.6, SD = sqrt(2267.2 / 4) = 23.81, z = -0.7: normal.
"""

from datetime import UTC, datetime, timedelta

import pytest

from capacity_radar.analytics.anomalies import find_anomalies, leave_one_out_z
from capacity_radar.analytics.metrics import WeekMetrics

WAITS = [30, 36, 40, 34, 90, 38]
FIRST_MONDAY = datetime(2026, 9, 7, tzinfo=UTC)


def week(
    i: int, *, arrivals=20, served=10, wait: float | None = 40.0, partial=False
) -> WeekMetrics:
    start = FIRST_MONDAY + timedelta(weeks=i)
    return WeekMetrics(
        start=start,
        end=start + timedelta(weeks=1),
        partial=partial,
        arrivals=arrivals,
        served=served,
        exited_unreviewed=5,
        queue_at_end=100,
        capacity_limited=True,
        median_wait_hours=wait,
    )


def test_scores_of_the_six_weeks():
    scores = leave_one_out_z(WAITS)
    w5 = scores[4]
    assert w5.others_mean == pytest.approx(35.6)
    assert w5.others_sd == pytest.approx(3.847, abs=0.001)
    assert w5.z == pytest.approx(14.1, abs=0.05)

    w1 = scores[0]
    assert (w1.others_mean, w1.others_sd) == (pytest.approx(47.6), pytest.approx(23.81, abs=0.01))
    assert [round(score.z, 1) for score in scores] == [-0.7, -0.4, -0.2, -0.5, 14.1, -0.3]


def test_only_the_spike_is_an_anomaly():
    weeks = [week(i, wait=wait) for i, wait in enumerate(WAITS)]
    [anomaly] = find_anomalies(weeks)
    assert (anomaly.metric, anomaly.week_start, anomaly.value) == (
        "median_wait_hours",
        FIRST_MONDAY + timedelta(weeks=4),
        90,
    )
    assert anomaly.z == pytest.approx(14.1, abs=0.05)


def test_unusually_low_weeks_are_found_too():
    served = [10, 11, 9, 10, 0, 12]  # nobody reviewed in week 5
    [anomaly] = find_anomalies([week(i, served=s) for i, s in enumerate(served)])
    assert anomaly.metric == "served" and anomaly.z < -3


def test_most_unusual_first():
    arrivals = [20, 30, 18, 21, 19, 23]  # week 2: z = (30 - 20.2) / 1.92 = 5.1, below 14.1
    weeks = [
        week(i, wait=w, arrivals=a) for i, (w, a) in enumerate(zip(WAITS, arrivals, strict=True))
    ]
    assert [anomaly.metric for anomaly in find_anomalies(weeks)] == [
        "median_wait_hours",
        "arrivals",
    ]


def test_the_partial_week_is_left_out():
    weeks = [week(i) for i in range(6)] + [week(6, arrivals=2, partial=True)]  # short week
    assert find_anomalies(weeks) == []


def test_weeks_without_a_median_wait_are_left_out_of_that_metric():
    weeks = [week(i, wait=wait) for i, wait in enumerate(WAITS)] + [week(6, wait=None)]
    assert [anomaly.metric for anomaly in find_anomalies(weeks)] == ["median_wait_hours"]


def test_identical_other_weeks_cannot_be_scored():
    assert leave_one_out_z([5, 5, 5, 5, 9])[4] is None  # SD of the others is 0


def test_too_few_weeks_to_judge():
    assert leave_one_out_z([30, 36, 90, 38]) == [None] * 4  # only 3 others each
