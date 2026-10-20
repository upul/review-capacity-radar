from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    MetaData,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

NAMING_CONVENTION = {
    "pk": "pk_%(table_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
}

SYNC_STATUSES = ("queued", "running", "succeeded", "failed", "skipped")


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map = {
        datetime: DateTime(timezone=True),  # every datetime column is timestamptz
        dict[str, Any]: JSONB,
    }


class Repository(Base):
    __tablename__ = "repository"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    owner: Mapped[str] = mapped_column(Text)  # stored lowercase; GitHub names are case-insensitive
    name: Mapped[str] = mapped_column(Text)

    __table_args__ = (UniqueConstraint("owner", "name"),)


class PullRequest(Base):
    __tablename__ = "pull_request"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    repo_id: Mapped[int] = mapped_column(ForeignKey("repository.id", ondelete="CASCADE"))
    number: Mapped[int]
    author_login: Mapped[str] = mapped_column(Text)  # "ghost" when the account was deleted
    author_is_bot: Mapped[bool]  # kept, not dropped, so bot share can be reported
    is_draft: Mapped[bool]
    created_at: Mapped[datetime]
    merged_at: Mapped[datetime | None]
    closed_at: Mapped[datetime | None]
    updated_at: Mapped[datetime]

    __table_args__ = (
        UniqueConstraint("repo_id", "number"),
        Index(None, "repo_id", "created_at"),
    )


class Review(Base):
    __tablename__ = "review"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    pr_id: Mapped[int] = mapped_column(ForeignKey("pull_request.id", ondelete="CASCADE"))
    reviewer_login: Mapped[str] = mapped_column(Text)
    reviewer_is_bot: Mapped[bool]
    # APPROVED, CHANGES_REQUESTED, COMMENTED, DISMISSED or PENDING.
    state: Mapped[str] = mapped_column(Text)
    submitted_at: Mapped[datetime | None]  # PENDING reviews have no submission time

    __table_args__ = (Index(None, "pr_id", "submitted_at"),)


class SyncState(Base):
    __tablename__ = "sync_state"

    repo_id: Mapped[int] = mapped_column(
        ForeignKey("repository.id", ondelete="CASCADE"), primary_key=True
    )
    cursor_updated_at: Mapped[datetime | None]  # newest PR updated_at seen
    history_start: Mapped[datetime | None]  # data is complete from here on (set on first run)
    list_etag: Mapped[str | None] = mapped_column(Text)  # ETag of the first PR-list page
    last_success_at: Mapped[datetime | None]


class SyncRun(Base):
    __tablename__ = "sync_run"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    owner: Mapped[str] = mapped_column(Text)
    name: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, server_default="queued")
    requested_at: Mapped[datetime] = mapped_column(server_default=func.now())
    started_at: Mapped[datetime | None]
    finished_at: Mapped[datetime | None]
    prs_upserted: Mapped[int] = mapped_column(server_default="0")
    error: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint(f"status IN {SYNC_STATUSES}", name="status"),
        Index(None, "status", "requested_at"),  # the background task claims oldest queued
    )


class NarrativeCache(Base):
    __tablename__ = "narrative_cache"

    cache_key: Mapped[str] = mapped_column(Text, primary_key=True)
    repo_id: Mapped[int] = mapped_column(ForeignKey("repository.id", ondelete="CASCADE"))
    prompt_version: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(Text)
    response: Mapped[dict[str, Any]]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
