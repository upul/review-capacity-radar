"""Throwaway exploration: look at real GitHub data before building the sync.

Not part of the app. Uses about 11 API requests.

    uv run python scripts/explore_github.py                      # scikit-learn/scikit-learn
    uv run python scripts/explore_github.py vllm-project/vllm
"""

import asyncio
import sys
from collections import Counter

from capacity_radar.github.client import GitHubClient
from capacity_radar.settings import get_settings

PR_LIMIT = 100  # one page
REVIEW_SAMPLE = 50  # PRs whose reviews we fetch


async def main(slug: str) -> None:
    owner, repo = slug.split("/")
    async with GitHubClient(get_settings().github_token) as gh:
        prs = []
        async for pr in gh.list_pull_requests(owner, repo):
            prs.append(pr)
            if len(prs) == PR_LIMIT:
                break  # stops paging: no further requests

        merged = sum(pr.merged_at is not None for pr in prs)
        closed_unmerged = sum(pr.closed_at is not None and pr.merged_at is None for pr in prs)
        print(f"\n=== {slug}: {len(prs)} most recently updated PRs")
        print(f"updated between {prs[-1].updated_at:%Y-%m-%d} and {prs[0].updated_at:%Y-%m-%d}")
        still_open = len(prs) - merged - closed_unmerged
        drafts = sum(pr.draft for pr in prs)
        bot_prs = sum(pr.author_is_bot for pr in prs)
        print(f"open: {still_open}   merged: {merged}   closed without merge: {closed_unmerged}")
        print(f"drafts: {drafts}   bot authors: {bot_prs}")
        bots = Counter(pr.author_login for pr in prs if pr.author_is_bot)
        if bots:
            print(f"bot authors seen: {dict(bots)}")

        print(f"\n=== Reviews on {REVIEW_SAMPLE} human-authored PRs")
        print("  PR      reviews  states                           updated_at - latest review")
        all_states: Counter[str] = Counter()
        reviewer_bots = 0
        for pr in [p for p in prs if not p.author_is_bot][:REVIEW_SAMPLE]:
            reviews = await gh.list_reviews(owner, repo, pr.number)
            states = Counter(r.state for r in reviews)
            all_states.update(states)
            reviewer_bots += sum(r.reviewer_is_bot for r in reviews)
            submitted = [r.submitted_at for r in reviews if r.submitted_at]
            if submitted:
                gap = f"{(pr.updated_at - max(submitted)).total_seconds():>10.0f}s"
            else:
                gap = "  (no reviews)"
            print(f"  #{pr.number:<6} {len(reviews):>7}  {str(dict(states)):<32} {gap}")

        print(f"\nreview states overall: {dict(all_states)}   reviews by bots: {reviewer_bots}")
        print("\nOpen question from Fig. 2: does a new review change the PR's updated_at?")
        print("  A gap of ~0s means the PR's last change was that review: evidence that it does.")
        print("  A negative gap would mean a review newer than updated_at: it does not.")
        reset = f"{gh.rate_limit_reset:%H:%M} UTC" if gh.rate_limit_reset else "unknown"
        print(f"\nrate limit: {gh.rate_limit_remaining} requests left, resets {reset}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "scikit-learn/scikit-learn"))
