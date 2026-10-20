"""Metric maths on a small example worked out by hand.

Window: Mon 5 Jan - Sun 18 Jan 2026 (14 days, two full weeks). Six eligible PRs:

  PR  created        first review    closed / merged   path out of the queue
  1   5 Jan 10:00    6 Jan 10:00     -                 reviewed after 24 h        (week 1)
  2   6 Jan 00:00    13 Jan 00:00    -                 reviewed after 168 h       (week 2)
  3   7 Jan 00:00    -               closed 9 Jan      exited unreviewed, closed  (week 1)
  4   30 Dec 2025    -               -                 still waiting at the end
  5   15 Jan 00:00   -               merged 16 Jan     exited unreviewed, merged  (week 2)
  6   10 Jan 00:00   12 Jan 00:00    closed 11 Jan     closed BEFORE its review: exited unreviewed
"""

from datetime import UTC, datetime

import pytest

from capacity_radar.analytics.metrics import PrFacts, capacity_report, reviewer_concentration
from capacity_radar.analytics.periods import Window


def at(day: int, hour: int = 0, month: int = 1, year: int = 2026) -> datetime:
    return datetime(year, month, day, hour, tzinfo=UTC)


WINDOW = Window(start=at(5), end=at(19))
PRS = [
    PrFacts(1, created_at=at(5, 10), first_review_at=at(6, 10), merged_at=None, closed_at=None),
    PrFacts(2, created_at=at(6), first_review_at=at(13), merged_at=None, closed_at=None),
    PrFacts(3, created_at=at(7), first_review_at=None, merged_at=None, closed_at=at(9)),
    PrFacts(
        4,
        created_at=at(30, month=12, year=2025),
        first_review_at=None,
        merged_at=None,
        closed_at=None,
    ),
    PrFacts(5, created_at=at(15), first_review_at=None, merged_at=at(16), closed_at=at(16)),
    PrFacts(6, created_at=at(10), first_review_at=at(12), merged_at=None, closed_at=at(11)),
]


@pytest.fixture(scope="module")
def report():
    return capacity_report(PRS, WINDOW)


def test_window_totals(report):
    assert report.arrivals == 5  # PRs 1, 2, 3, 5, 6 (PR 4 was created before the window)
    assert report.served == 2  # PRs 1 and 2
    assert report.exited_unreviewed_closed == 2  # PR 3, and PR 6 (closed before its review)
    assert report.exited_unreviewed_merged == 1  # PR 5


def test_wait_times(report):
    assert report.wait_sample_size == 2
    assert report.wait_hours_median == pytest.approx(96)  # (24 + 168) / 2
    assert report.wait_hours_p90 == pytest.approx(153.6)  # 24 + 0.9 * (168 - 24)


def test_queue_and_still_waiting(report):
    assert report.queue_at_start == 1  # only PR 4 (PR 1 arrives at 10:00)
    assert report.queue_at_end == 1  # PR 4 is still waiting
    assert report.still_waiting_median_age_hours == pytest.approx(20 * 24)  # 30 Dec -> 19 Jan


def test_weekly_series(report):
    week1, week2 = report.weeks
    assert (week1.arrivals, week1.served, week1.exited_unreviewed) == (4, 1, 2)
    assert week1.queue_at_end == 2  # PRs 2 and 4 at Mon 12 Jan 00:00
    assert week1.median_wait_hours == pytest.approx(24)
    assert (week2.arrivals, week2.served, week2.exited_unreviewed) == (1, 1, 1)
    assert week2.queue_at_end == 1
    assert week2.median_wait_hours == pytest.approx(168)
    assert week1.capacity_limited and week2.capacity_limited  # the queue never emptied
    assert not week1.partial and not week2.partial


def test_littles_law(report):
    law = report.littles_law
    assert law.arrival_rate_per_day == pytest.approx(5 / 14)
    # PRs that left the queue in the window, in days: 1, 7, 2, 1 (PR 5), 1 (PR 6)
    assert law.mean_days_in_queue == pytest.approx(12 / 5)
    assert law.predicted_queue == pytest.approx(5 / 14 * 12 / 5)
    # Daily counts at each midnight from 6 to 19 Jan: PR 4 on all 14 days, PR 2 on 7,
    # PR 3 on 2, PRs 1, 5 and 6 on 1 each: 26 in total
    assert law.measured_queue == pytest.approx(26 / 14)
    # Far above prediction: the long-waiting PR 4 makes this queue anything but steady
    assert law.gap_percent > 100


def test_partial_last_week():
    weeks = Window(start=at(5), end=at(15)).buckets()  # 10 days
    assert [w.partial for w in weeks] == [False, True]


def test_reviewer_concentration_counts_distinct_prs():
    # alice replied twice on PR 1 and reviewed PR 2; bob reviewed PR 3
    pairs = [("alice", 1), ("alice", 1), ("alice", 2), ("bob", 3)]
    result = reviewer_concentration(pairs)
    assert result["reviewers"] == 2
    assert result["top"][0] == {"login": "alice", "prs_reviewed": 2}
    assert result["top1_share"] == pytest.approx(2 / 3)
    assert result["top3_share"] == pytest.approx(1.0)


def test_empty_inputs_do_not_crash():
    report = capacity_report([], WINDOW)
    assert report.arrivals == 0 and report.wait_hours_median is None
    assert report.littles_law.predicted_queue is None
    assert reviewer_concentration([])["top1_share"] is None
