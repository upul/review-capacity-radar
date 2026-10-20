"""The analysis window and its weekly buckets.

Weeks are 7-day periods counted from the window's first day, so a 12-week window has
exactly 12 buckets. If the window doesn't divide into whole weeks, the last bucket is
shorter and marked partial. All times are UTC.
"""

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

DEFAULT_WEEKS = 12


class WindowError(ValueError):
    """The requested window is invalid or outside the synced data."""


@dataclass(frozen=True)
class Bucket:
    start: datetime  # inclusive
    end: datetime  # exclusive
    partial: bool


@dataclass(frozen=True)
class Window:
    start: datetime  # inclusive: 00:00 UTC on the first day
    end: datetime  # exclusive: 00:00 UTC on the day after the last day

    @property
    def days(self) -> float:
        return (self.end - self.start) / timedelta(days=1)

    def buckets(self) -> list[Bucket]:
        buckets, start = [], self.start
        while start < self.end:
            end = min(start + timedelta(weeks=1), self.end)
            buckets.append(Bucket(start, end, partial=end - start < timedelta(weeks=1)))
            start = end
        return buckets


def make_window(
    first_day: date | None,
    last_day: date | None,
    *,
    history_start: datetime,
    data_as_of: datetime,
    max_days: int,
) -> Window:
    """Validate a requested date range against the synced data.

    Defaults: the last day with data, and the 12 weeks ending on it.
    """
    last_day = last_day or data_as_of.date()
    first_day = first_day or last_day - timedelta(weeks=DEFAULT_WEEKS) + timedelta(days=1)

    if first_day > last_day:
        raise WindowError("'from' must be on or before 'to'")
    if (last_day - first_day).days + 1 > max_days:
        raise WindowError(f"the window can be at most {max_days} days")

    window = Window(_midnight(first_day), _midnight(last_day + timedelta(days=1)))
    if window.start < history_start:
        first_full_day = history_start.date() + timedelta(days=history_start.time() != time.min)
        raise WindowError(f"synced data covers {first_full_day} onwards; move 'from' later")
    if first_day > data_as_of.date():
        raise WindowError(f"no data after {data_as_of.date()}; the last sync was then")
    return window


def _midnight(day: date) -> datetime:
    return datetime.combine(day, time.min, tzinfo=UTC)
