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

    Defaults: the last complete day of data, and the 12 weeks ending on it (shortened if
    the synced history doesn't reach back that far).
    """
    first_full_day = _first_full_day(history_start)
    last_day = last_day or (data_as_of - timedelta(days=1)).date()  # last complete day
    if first_day is None:
        twelve_weeks_back = last_day - timedelta(weeks=DEFAULT_WEEKS) + timedelta(days=1)
        first_day = max(twelve_weeks_back, first_full_day)

    if first_day > last_day:
        raise WindowError("'from' must be on or before 'to'")
    if (last_day - first_day).days + 1 > max_days:
        raise WindowError(f"the window can be at most {max_days} days")

    window = Window(_midnight(first_day), _midnight(last_day + timedelta(days=1)))
    if first_day < first_full_day:
        raise WindowError(f"synced data covers {first_full_day} onwards; move 'from' later")
    if first_day > data_as_of.date():
        raise WindowError(f"no data after {data_as_of.date()}; the last sync was then")
    return window


def recent_full_weeks(*, history_start: datetime, data_as_of: datetime, weeks: int) -> Window:
    """The latest `weeks` complete weeks of data, ending at the last complete day. Fewer
    weeks if the synced history is shorter; every bucket of the result is a full week."""
    end = _midnight(data_as_of.date())  # exclusive: the day of data_as_of is incomplete
    available = (end - _midnight(_first_full_day(history_start))) // timedelta(weeks=1)
    return Window(end - timedelta(weeks=max(min(weeks, available), 0)), end)


def _first_full_day(history_start: datetime) -> date:
    """The first day the synced data covers completely."""
    starts_at_midnight = history_start.time() == time.min
    return history_start.date() + timedelta(days=0 if starts_at_midnight else 1)


def _midnight(day: date) -> datetime:
    return datetime.combine(day, time.min, tzinfo=UTC)
