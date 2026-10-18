from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, HttpUrl, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PositiveInt = Annotated[int, Field(gt=0)]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,  # "LLM_BASE_URL=" means unset; an empty token means missing
        extra="ignore",
    )

    #  Runtime
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    #  Database
    # Secret because the URL carries the password. Must use the asyncpg driver.
    database_url: SecretStr

    #  GitHub
    github_token: SecretStr
    # Repos the scheduled background sync keeps fresh. Comma-separated in the environment:
    sync_repos: Annotated[list[str], NoDecode] = [
        "scikit-learn/scikit-learn",
        "vllm-project/vllm",
    ]
    github_max_concurrency: Annotated[int, Field(ge=1, le=10)] = 4
    github_timeout_seconds: PositiveInt = 30

    # --- Background sync ---
    sync_enabled: bool = True  # tests turn this off
    sync_interval_minutes: PositiveInt = 30  # scheduled incremental sync of sync_repos
    sync_poll_seconds: PositiveInt = 5  # how often queued runs are picked up
    sync_lookback_weeks: Annotated[int, Field(ge=1, le=52)] = 12

    # --- LLM (OpenAI, or OpenRouter via llm_base_url) ---
    llm_api_key: SecretStr
    llm_model: str = Field(min_length=1)
    llm_base_url: HttpUrl | None = None  # None means the OpenAI default endpoint
    llm_timeout_seconds: PositiveInt = 60
    llm_max_retries: Annotated[int, Field(ge=0, le=5)] = 2

    #  API limits
    max_window_days: Annotated[int, Field(ge=7, le=366)] = 182  # longest date range
    max_forecast_weeks: Annotated[int, Field(ge=1, le=26)] = 12  # longest forecast horizon

    @field_validator("database_url")
    @classmethod
    def _require_asyncpg(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().startswith("postgresql+asyncpg://"):
            raise ValueError("DATABASE_URL must start with 'postgresql+asyncpg://'")
        return value

    @field_validator("sync_repos", mode="before")
    @classmethod
    def _split_repos(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator("sync_repos")
    @classmethod
    def _check_repos(cls, value: list[str]) -> list[str]:
        for slug in value:
            owner, _, repo = slug.partition("/")
            if not owner or not repo or "/" in repo:
                raise ValueError(f"expected 'owner/repo', got {slug!r}")
        return list(dict.fromkeys(value))  # drop duplicates, keep order

    @field_validator("llm_base_url")
    @classmethod
    def _require_https(cls, value: HttpUrl | None) -> HttpUrl | None:
        if value is not None and value.scheme != "https":
            raise ValueError("LLM_BASE_URL must use https")
        return value


@lru_cache
def get_settings() -> Settings:
    """Read settings once per process. Tests override it via FastAPI dependency_overrides."""
    return Settings()
