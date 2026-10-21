"""Writes the narrative: ask the LLM, check its answer, retry once with the problems, and
fall back to a plain template built from the facts if it still fails. Validated answers are
cached, so the same facts with the same prompt and model never cost a second LLM call."""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, replace
from typing import Literal

from openai import OpenAIError
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from capacity_radar.analytics.metrics import CapacityReport
from capacity_radar.analytics.queries import DataQuality
from capacity_radar.db.models import NarrativeCache
from capacity_radar.narrative.confidence import capped_confidence
from capacity_radar.narrative.evidence import Fact
from capacity_radar.narrative.llm import PROMPT_VERSION, NarrativeLLM
from capacity_radar.narrative.validator import check, citations

logger = logging.getLogger(__name__)

ATTEMPTS = 2  # the first draft, and one retry that is told what was wrong
FALLBACK_CONFIDENCE = 0.3


@dataclass(frozen=True)
class Narrative:
    summary: str
    hypothesis: str
    evidence: list[str]  # fact IDs, in the order they were first cited
    confidence: float
    confidence_notes: list[str]
    source: Literal["llm", "cache", "fallback"]
    problems: list[str]  # why the LLM's drafts were rejected; only set for the fallback


async def write_narrative(
    llm: NarrativeLLM, facts: list[Fact], report: CapacityReport, quality: DataQuality
) -> Narrative:
    feedback = None
    problems: list[str] = []
    for _ in range(ATTEMPTS):
        try:
            raw = await llm.draft(facts, feedback)
        except OpenAIError as exc:  # the SDK has already retried what is worth retrying
            logger.warning("LLM call failed: %s", type(exc).__name__)
            return fallback(facts, [f"the LLM call failed ({type(exc).__name__})"])
        except ValueError:
            raw, problems = {}, ["the answer was not a JSON object"]
        else:
            checked = check(raw, facts)
            if checked.draft is not None:
                draft = checked.draft
                confidence = capped_confidence(draft.confidence, report, quality)
                return Narrative(
                    summary=draft.summary,
                    hypothesis=draft.hypothesis,
                    evidence=_cited(draft.summary, draft.hypothesis, draft.evidence),
                    confidence=confidence.value,
                    confidence_notes=confidence.notes,
                    source="llm",
                    problems=[],
                )
            problems = checked.problems
        logger.warning("narrative draft rejected: %s", "; ".join(problems))
        feedback = (
            f"Your previous answer was:\n{json.dumps(raw)}\nIt was rejected because:\n- "
            + "\n- ".join(problems)
            + "\nReturn a corrected answer as JSON, following all the rules."
        )
    return fallback(facts, problems)


def fallback(facts: list[Fact], problems: list[str]) -> Narrative:
    """A plain, always-true narrative: the first facts as written, and no hypothesis."""
    first, second = facts[0], facts[1]  # the window's flow and queue facts are always present
    return Narrative(
        summary=f"{first.text} [{first.id}]. {second.text} [{second.id}].",
        hypothesis="No explanation could be validated, so none is offered; "
        "see the facts for the computed evidence.",
        evidence=[first.id, second.id],
        confidence=FALLBACK_CONFIDENCE,
        confidence_notes=["fixed: no validated LLM explanation"],
        source="fallback",
        problems=problems,
    )


def _cited(summary: str, hypothesis: str, listed: list[str]) -> list[str]:
    """Fact IDs cited in the text or listed as evidence, each once, in order of appearance."""
    return list(dict.fromkeys([*citations(summary), *citations(hypothesis), *listed]))


# --- cache ---


def cache_key(repo_id: int, model: str, facts: list[Fact]) -> str:
    """Same repository, facts, prompt and model: same narrative."""
    payload = [repo_id, PROMPT_VERSION, model, [fact.text for fact in facts]]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


async def cached(session: AsyncSession, key: str) -> Narrative | None:
    response = await session.scalar(
        select(NarrativeCache.response).where(NarrativeCache.cache_key == key)
    )
    return replace(Narrative(**response), source="cache") if response else None


async def store(
    session: AsyncSession, key: str, repo_id: int, model: str, narrative: Narrative
) -> None:
    """Keep validated LLM answers only; a fallback is retried on the next request."""
    if narrative.source != "llm":
        return
    await session.execute(
        insert(NarrativeCache)
        .values(
            cache_key=key,
            repo_id=repo_id,
            prompt_version=PROMPT_VERSION,
            model=model,
            response=asdict(narrative),
        )
        .on_conflict_do_nothing(index_elements=[NarrativeCache.cache_key])
    )
    await session.commit()
