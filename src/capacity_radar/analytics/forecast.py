"""Monte Carlo forecast of the review queue.

Each simulated future replays randomly chosen past weeks: every future week copies the net
change (arrivals - served - exited unreviewed) of one past week, picked at random. Whole
weeks are copied, so busy and quiet weeks keep their realistic mix of arrivals and reviews.
Many such futures together give a range: p10 (a good case), p50 (the middle), p90 (a bad
case), and the share of futures in which the queue ends bigger than it is today.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

RUNS = 2000  # simulated futures; enough for stable percentiles, still fast
SEED = 0  # fixed, so the same data always gives the same forecast
MIN_HISTORY_WEEKS = 4  # fewer past weeks than this is too little to replay


@dataclass(frozen=True)
class ForecastWeek:
    week: int  # 1 = the week after the history ends
    p10: float
    p50: float
    p90: float
    chance_queue_grows: float  # share of futures where the queue is bigger than at the start


def forecast_queue(
    start_queue: int, net_changes: Sequence[int], weeks: int, *, runs: int = RUNS, seed: int = SEED
) -> list[ForecastWeek]:
    return summarize(simulate(start_queue, net_changes, weeks, runs=runs, seed=seed), start_queue)


def simulate(
    start_queue: int, net_changes: Sequence[int], weeks: int, *, runs: int, seed: int
) -> np.ndarray:
    """Queue size after each future week: one row per simulated future, one column per week."""
    if not net_changes:
        raise ValueError("need at least one past week to replay")
    rng = np.random.default_rng(seed)
    picks = rng.choice(np.asarray(net_changes), size=(runs, weeks))  # a past week per cell
    queues = np.empty((runs, weeks))
    queue = np.full(runs, start_queue)
    for week in range(weeks):
        queue = np.maximum(queue + picks[:, week], 0)  # a queue can't go below empty
        queues[:, week] = queue
    return queues


def summarize(queues: np.ndarray, start_queue: int) -> list[ForecastWeek]:
    """Percentiles across the simulated futures, week by week."""
    p10, p50, p90 = np.percentile(queues, [10, 50, 90], axis=0)
    grows = (queues > start_queue).mean(axis=0)
    return [
        ForecastWeek(
            week + 1, float(p10[week]), float(p50[week]), float(p90[week]), float(grows[week])
        )
        for week in range(queues.shape[1])
    ]
