"""Background sync: keeps the repositories in SYNC_REPOS fresh while the app runs."""

import asyncio
import logging

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from capacity_radar.github.client import GitHubClient
from capacity_radar.settings import Settings
from capacity_radar.sync.service import sync_repo

logger = logging.getLogger(__name__)


async def sync_forever(
    settings: Settings, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """Sync every repository, wait SYNC_INTERVAL_MINUTES, repeat. Runs until the app stops."""
    while True:
        async with GitHubClient(settings.github_token, settings.github_timeout_seconds) as gh:
            await sync_once(gh, session_factory, settings.sync_repos, settings.sync_lookback_weeks)
        await asyncio.sleep(settings.sync_interval_minutes * 60)


async def sync_once(
    gh: GitHubClient,
    session_factory: async_sessionmaker[AsyncSession],
    repos: list[str],
    lookback_weeks: int,
) -> None:
    """Sync each repository in turn. One failing doesn't stop the others; it is retried
    on the next cycle."""
    for slug in repos:
        owner, repo = slug.split("/")  # settings already checked the "owner/repo" form
        try:
            result = await sync_repo(gh, session_factory, owner, repo, lookback_weeks)
        except Exception:
            logger.exception("sync of %s failed; will retry next cycle", slug)
            continue
        logger.info("synced %s: %d PRs changed", slug, result.prs_synced)
