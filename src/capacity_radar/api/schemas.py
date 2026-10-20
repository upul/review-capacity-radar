"""Response models for the insights API. They define the JSON shape shown in /docs."""

from datetime import date, datetime

from pydantic import BaseModel, Field


class WindowOut(BaseModel):
    start: date = Field(serialization_alias="from")
    end: date = Field(serialization_alias="to")  # inclusive, like the request
    days: int


class WaitHours(BaseModel):
    median: float | None
    p90: float | None
    sample_size: int = Field(description="PRs first reviewed in the window")


class Queue(BaseModel):
    at_start: int
    at_end: int = Field(description="PRs still waiting for a first review when the window ends")
    still_waiting_median_age_hours: float | None


class ExitedUnreviewed(BaseModel):
    merged: int
    closed: int


class Summary(BaseModel):
    arrivals: int = Field(description="Eligible PRs created in the window")
    served: int = Field(description="PRs that got their first qualifying review in the window")
    exited_unreviewed: ExitedUnreviewed
    wait_hours: WaitHours
    queue: Queue


class LittlesLawOut(BaseModel):
    arrival_rate_per_day: float
    mean_days_in_queue: float | None
    predicted_queue: float | None = Field(description="arrival rate × mean days in queue")
    measured_queue: float = Field(description="average of daily queue counts")
    gap_percent: float | None = Field(
        description="measured vs predicted; a large gap means the queue isn't steady"
    )


class ReviewerShare(BaseModel):
    login: str
    prs_reviewed: int


class Reviewers(BaseModel):
    reviewers: int
    top1_share: float | None
    top3_share: float | None
    top: list[ReviewerShare]


class Week(BaseModel):
    start: date
    end: date  # inclusive
    partial: bool
    arrivals: int
    served: int
    exited_unreviewed: int
    queue_at_end: int
    capacity_limited: bool = Field(
        description="the queue never emptied, so 'served' estimates review capacity"
    )
    median_wait_hours: float | None


class DataQualityOut(BaseModel):
    history_start: datetime
    prs_created: int
    bot_pr_share: float | None
    drafts_excluded: int
    merged_with_review_share: float | None = Field(
        description="low values mean review happens outside GitHub reviews in this repo"
    )
    capacity_limited_weeks: int


class CapacityResponse(BaseModel):
    repo: str
    window: WindowOut
    data_as_of: datetime
    summary: Summary
    littles_law: LittlesLawOut
    reviewers: Reviewers
    weeks: list[Week]
    data_quality: DataQualityOut
