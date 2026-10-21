"""The forecast, checked on the hand-worked example: queue 10 today, three past weeks with
net changes +5, -8 and +10, forecast 2 weeks ahead. That gives 3 x 3 = 9 possible futures:

    W1,W1: 15 -> 20    W2,W1:  2 -> 7     W3,W1: 20 -> 25
    W1,W2: 15 -> 7     W2,W2:  2 -> 0     W3,W2: 20 -> 12
    W1,W3: 15 -> 25    W2,W3:  2 -> 12    W3,W3: 20 -> 30
                       (2 - 8 = -6, floored at 0)
"""

from collections import Counter
from datetime import UTC, datetime

import numpy as np
import pytest

from capacity_radar.analytics.forecast import forecast_queue, simulate, summarize
from capacity_radar.analytics.periods import recent_full_weeks

QUEUE_NOW = 10
NET_CHANGES = [5, -8, 10]
ALL_FUTURES = {(15, 20), (15, 7), (15, 25), (2, 7), (2, 0), (2, 12), (20, 25), (20, 12), (20, 30)}


def test_percentiles_of_the_nine_futures():
    queues = np.array(sorted(ALL_FUTURES))  # 9 rows (futures) x 2 columns (weeks)
    week1, week2 = summarize(queues, QUEUE_NOW)

    # Week 2 sorted: 0 7 7 12 12 20 25 25 30
    assert week2.p50 == 12  # position 0.5 x 8 = 4
    assert week2.p10 == pytest.approx(5.6)  # position 0.8: 0 + 0.8 x (7 - 0)
    assert week2.p90 == pytest.approx(26)  # position 7.2: 25 + 0.2 x (30 - 25)
    assert week2.chance_queue_grows == pytest.approx(6 / 9)  # 20 25 12 25 12 30 are above 10

    # Week 1 sorted: 2 2 2 15 15 15 20 20 20
    assert (week1.week, week1.p50, week1.chance_queue_grows) == (1, 15, pytest.approx(6 / 9))


def test_every_simulated_future_is_one_of_the_nine():
    queues = simulate(QUEUE_NOW, NET_CHANGES, weeks=2, runs=9000, seed=0)
    futures = Counter(tuple(int(q) for q in row) for row in queues)

    assert set(futures) == ALL_FUTURES
    for count in futures.values():  # each is equally likely: about 1000 of 9000
        assert 850 < count < 1150


def test_the_queue_never_goes_below_empty():
    queues = simulate(3, [-5], weeks=3, runs=10, seed=0)
    assert (queues == 0).all()


def test_identical_weeks_give_a_straight_line():
    forecast = forecast_queue(100, [4, 4, 4], weeks=3)
    assert [(week.p10, week.p50, week.p90) for week in forecast] == [
        (104, 104, 104),
        (108, 108, 108),
        (112, 112, 112),
    ]
    assert all(week.chance_queue_grows == 1 for week in forecast)


def test_same_data_same_forecast():
    assert forecast_queue(50, [3, -6, 9, 0], weeks=8) == forecast_queue(50, [3, -6, 9, 0], weeks=8)


def test_percentiles_are_ordered_and_the_range_widens():
    forecast = forecast_queue(200, [12, -15, 30, -4, 7, -20], weeks=6)
    assert all(week.p10 <= week.p50 <= week.p90 for week in forecast)
    assert forecast[-1].p90 - forecast[-1].p10 > forecast[0].p90 - forecast[0].p10


def test_no_history_is_an_error():
    with pytest.raises(ValueError):
        simulate(10, [], weeks=2, runs=10, seed=0)


def test_history_is_whole_weeks_ending_at_the_last_complete_day():
    # Data until 20 Oct 00:15; history fully covers 29 Jul onwards: 83 days, so 11 whole weeks
    window = recent_full_weeks(
        history_start=datetime(2026, 7, 28, 0, 15, tzinfo=UTC),
        data_as_of=datetime(2026, 10, 20, 0, 15, tzinfo=UTC),
        weeks=12,
    )
    assert (window.start, window.end) == (
        datetime(2026, 8, 4, tzinfo=UTC),
        datetime(2026, 10, 20, tzinfo=UTC),  # exclusive: the last day is 19 Oct
    )
    assert len(window.buckets()) == 11
    assert not any(bucket.partial for bucket in window.buckets())
