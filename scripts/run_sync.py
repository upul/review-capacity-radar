"""Run one sync by hand and report what it did. Not part of the app.

uv run python scripts/run_sync.py                      # scikit-learn/scikit-learn
uv run python scripts/run_sync.py vllm-project/vllm
"""

import asyncio
import logging
import sys
import time

from capacity_radar.db.engine import create_engine, create_session_factory, init_db
from capacity_radar.github.client import GitHubClient
from capacity_radar.settings import get_settings
from capacity_radar.sync.service import sync_repo


async def main(slug: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request is too noisy
    settings = get_settings()
    owner, repo = slug.split("/")

    engine = create_engine(settings)
    await init_db(engine)
    started = time.perf_counter()
    try:
        async with GitHubClient(settings.github_token, settings.github_timeout_seconds) as gh:
            result = await sync_repo(
                gh, create_session_factory(engine), owner, repo, settings.sync_lookback_weeks
            )
            remaining = gh.rate_limit_remaining
    finally:
        await engine.dispose()

    print(f"\n=== Synced {result.repo} in {time.perf_counter() - started:.1f}s")
    print(f"first run:       {result.first_run}")
    print(f"pull requests:   {result.prs_synced} synced, {result.prs_unchanged} unchanged")
    print(f"reviews:         {result.reviews_synced}")
    print(f"GitHub requests: {result.requests_made}  ({remaining} left this hour)")
    print(f"new cursor:      {result.cursor}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "scikit-learn/scikit-learn"))
