from datetime import UTC, date, datetime

import pytest

from capacity_radar.analytics.periods import WindowError, make_window

HISTORY_START = datetime(2026, 7, 27, 23, 27, tzinfo=UTC)
DATA_AS_OF = datetime(2026, 10, 19, 23, 55, tzinfo=UTC)


def window(first=None, last=None, max_days=182):
    return make_window(
        first, last, history_start=HISTORY_START, data_as_of=DATA_AS_OF, max_days=max_days
    )


def test_default_is_the_last_twelve_weeks():
    w = window()
    assert w.start == datetime(2026, 7, 28, tzinfo=UTC)
    assert w.end == datetime(2026, 10, 20, tzinfo=UTC)  # exclusive: the day after 19 Oct
    assert len(w.buckets()) == 12 and not any(b.partial for b in w.buckets())


def test_explicit_window():
    w = window(date(2026, 9, 1), date(2026, 9, 30))
    assert w.days == 30 and len(w.buckets()) == 5  # 4 full weeks + 2 days


@pytest.mark.parametrize(
    ("first", "last", "message"),
    [
        (date(2026, 9, 30), date(2026, 9, 1), "on or before"),
        (date(2026, 7, 1), date(2026, 9, 1), "covers 2026-07-28 onwards"),
        (date(2026, 11, 1), date(2026, 11, 5), "no data after"),
    ],
)
def test_invalid_windows(first, last, message):
    with pytest.raises(WindowError, match=message):
        window(first, last)


def test_first_full_day_is_accepted():
    # history starts 27 Jul 23:27, so 28 Jul is the first fully covered day
    assert window(date(2026, 7, 28), date(2026, 8, 31)).start == datetime(2026, 7, 28, tzinfo=UTC)


def test_window_too_long():
    with pytest.raises(WindowError, match="at most 30 days"):
        window(date(2026, 8, 1), date(2026, 9, 30), max_days=30)
