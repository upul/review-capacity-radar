"""Read-only client for the GitHub REST API: pull requests and their reviews.

Requests can only ever go to api.github.com: the base URL is a constant, owner and repo
names are escaped before they reach a URL path, and pagination links are checked before
they are followed.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any, Literal, Self
from urllib.parse import quote

import httpx
from pydantic import BaseModel, SecretStr

logger = logging.getLogger(__name__)

GITHUB_API_URL = "https://api.github.com"
PAGE_SIZE = 100  # GitHub's maximum
RATE_LIMIT_FLOOR = 50  # stop with requests to spare, so other work using the token still runs

# Transient failures (timeouts, dropped connections, GitHub 5xx) are retried with growing
# waits of 1, 2 and 4 seconds. Over thousands of requests an occasional blip is normal.
MAX_ATTEMPTS = 4
FIRST_BACKOFF_SECONDS = 1
RETRYABLE_STATUSES = {500, 502, 503, 504}
MAX_RETRY_AFTER_SECONDS = 60  # longest "slow down" wait we accept before giving up


class GitHubError(Exception):
    """A request to GitHub failed."""


class RateLimitExhausted(GitHubError):
    """The token's request budget is (nearly) spent until `reset_at`."""

    def __init__(self, reset_at: datetime | None) -> None:
        self.reset_at = reset_at
        super().__init__(f"GitHub rate limit nearly exhausted; resets at {reset_at}")


# --- Response models: only the fields we use. Unknown fields are ignored. ---


class GitHubUser(BaseModel):
    login: str
    type: str  # "User", "Bot" or "Organization"

    @property
    def is_bot(self) -> bool:
        return self.type == "Bot" or self.login.endswith("[bot]")


class GitHubRepository(BaseModel):
    id: int
    name: str
    owner: GitHubUser


class PullRequest(BaseModel):
    id: int
    number: int
    user: GitHubUser | None  # None when the author's account was deleted
    draft: bool = False
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None
    merged_at: datetime | None

    @property
    def author_login(self) -> str:
        return self.user.login if self.user else "ghost"  # GitHub's own name for deleted users

    @property
    def author_is_bot(self) -> bool:
        return self.user.is_bot if self.user else False


class Review(BaseModel):
    id: int
    user: GitHubUser | None
    state: str  # APPROVED, CHANGES_REQUESTED, COMMENTED, DISMISSED or PENDING
    submitted_at: datetime | None = None  # absent for PENDING reviews

    @property
    def reviewer_login(self) -> str:
        return self.user.login if self.user else "ghost"

    @property
    def reviewer_is_bot(self) -> bool:
        return self.user.is_bot if self.user else False


# --- Client ---


class GitHubClient:
    """Use as `async with GitHubClient(token) as gh:` so the connection pool is closed."""

    def __init__(self, token: SecretStr, timeout_seconds: float = 30) -> None:
        self._http = httpx.AsyncClient(
            base_url=GITHUB_API_URL,
            headers={
                "Authorization": f"Bearer {token.get_secret_value()}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "review-capacity-radar",
            },
            timeout=timeout_seconds,
        )
        self.rate_limit_remaining: int | None = None  # from the latest response
        self.rate_limit_reset: datetime | None = None
        self.requests_made = 0  # requests sent by this client, for reporting

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._http.aclose()

    async def get_repository(self, owner: str, repo: str) -> GitHubRepository:
        """The repository's GitHub ID and its canonical owner and name."""
        response = await self._get(f"/repos/{_segment(owner)}/{_segment(repo)}", None)
        return GitHubRepository.model_validate(response.json())

    async def list_pull_requests(
        self, owner: str, repo: str, state: Literal["all", "open"] = "all"
    ) -> AsyncIterator[PullRequest]:
        """Pull requests, most recently updated first. Stop iterating early to save
        requests: pages are fetched only as the caller consumes them."""
        path = f"/repos/{_segment(owner)}/{_segment(repo)}/pulls"
        params = {"state": state, "sort": "updated", "direction": "desc", "per_page": PAGE_SIZE}
        async for item in self._paginate(path, params):
            yield PullRequest.model_validate(item)

    async def list_reviews(self, owner: str, repo: str, number: int) -> list[Review]:
        """Every review on one pull request, oldest first."""
        path = f"/repos/{_segment(owner)}/{_segment(repo)}/pulls/{number}/reviews"
        return [
            Review.model_validate(item)
            async for item in self._paginate(path, {"per_page": PAGE_SIZE})
        ]

    async def _paginate(
        self, path: str, params: dict[str, Any] | None
    ) -> AsyncIterator[dict[str, Any]]:
        url: str | None = path
        while url is not None:
            response = await self._get(url, params)
            for item in response.json():
                yield item
            url = _next_page_url(response)
            params = None  # the next-page URL already carries the query string

    async def _get(self, url: str, params: dict[str, Any] | None) -> httpx.Response:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if (
                self.rate_limit_remaining is not None
                and self.rate_limit_remaining < RATE_LIMIT_FLOOR
            ):
                raise RateLimitExhausted(self.rate_limit_reset)

            error: Exception | None = None
            try:
                self.requests_made += 1
                response = await self._http.get(url, params=params)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:  # transient
                problem, error = type(exc).__name__, exc
                delay = FIRST_BACKOFF_SECONDS * 2 ** (attempt - 1)
            except httpx.HTTPError as exc:  # anything else will not fix itself
                raise GitHubError(f"GET {url} failed: {type(exc).__name__}") from exc
            else:
                self._record_rate_limit(response)
                logger.debug(
                    "GET %s -> %s (rate limit remaining: %s)",
                    response.url.path,
                    response.status_code,
                    self.rate_limit_remaining,
                )
                if response.status_code in (403, 429) and self.rate_limit_remaining == 0:
                    raise RateLimitExhausted(self.rate_limit_reset)
                if response.is_success:  # 2xx only: a redirect is reported, not followed
                    return response
                retry_delay = _retry_delay(response, attempt)
                if retry_delay is None:  # e.g. 404, or 403 for a permission problem
                    raise GitHubError(f"GET {response.url.path} returned {response.status_code}")
                problem, delay = f"HTTP {response.status_code}", retry_delay

            if attempt == MAX_ATTEMPTS:
                raise GitHubError(
                    f"GET {url} failed after {MAX_ATTEMPTS} attempts: {problem}"
                ) from error
            logger.warning(
                "GET %s: %s, retrying in %ss (attempt %d of %d)",
                url,
                problem,
                delay,
                attempt,
                MAX_ATTEMPTS,
            )
            await asyncio.sleep(delay)
        raise AssertionError("unreachable")  # the loop always returns or raises

    def _record_rate_limit(self, response: httpx.Response) -> None:
        remaining = response.headers.get("X-RateLimit-Remaining")
        reset = response.headers.get("X-RateLimit-Reset")
        if remaining is not None:
            self.rate_limit_remaining = int(remaining)
        if reset is not None:
            self.rate_limit_reset = datetime.fromtimestamp(int(reset), tz=UTC)


def _retry_delay(response: httpx.Response, attempt: int) -> float | None:
    """Seconds to wait before retrying this error response, or None if retrying won't help."""
    if response.status_code in (403, 429):
        # GitHub's "slow down" (secondary rate limit) says how long to wait. A 403 without
        # Retry-After is a permission problem, and a long wait is not worth blocking on.
        retry_after = response.headers.get("Retry-After", "")
        if retry_after.isdigit() and int(retry_after) <= MAX_RETRY_AFTER_SECONDS:
            return int(retry_after)
        return None
    if response.status_code in RETRYABLE_STATUSES:
        return FIRST_BACKOFF_SECONDS * 2 ** (attempt - 1)
    return None


def _segment(name: str) -> str:
    """Escape an owner or repo name for use as one URL path segment."""
    if not name or name in {".", ".."}:
        raise ValueError(f"invalid GitHub name: {name!r}")
    return quote(name, safe="")  # "/" and other special characters cannot change the path


def _next_page_url(response: httpx.Response) -> str | None:
    """The `rel="next"` URL from GitHub's Link header, if it points back at the API."""
    url = response.links.get("next", {}).get("url")
    if url is None:
        return None
    if not url.startswith(GITHUB_API_URL + "/"):
        raise GitHubError("refusing to follow a pagination link outside api.github.com")
    return url
