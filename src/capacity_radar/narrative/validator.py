"""Checks an LLM draft before anyone sees it.

A draft passes only if it has the right shape, cites facts that exist, and every number it
writes appears in a fact cited in the same sentence. That makes the evidence chain real:
each number can be traced to the fact it came from, and an invented number is rejected.
"""

import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from capacity_radar.narrative.evidence import Fact

MIN_EVIDENCE = 2
CITATION = re.compile(r"\[(F\d+(?:\s*,\s*F\d+)*)\]")  # [F2], or several: [F2, F3]
DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")  # dates are labels, not numbers to check
NAMED_NUMBER = re.compile(r"\b[pF]\d+\b")  # p50, p90 and fact IDs are names
NUMBER = re.compile(r"\d+(?:\.\d+)?")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class Draft(BaseModel):
    summary: str = Field(min_length=1, max_length=600)
    hypothesis: str = Field(min_length=1, max_length=600)
    evidence: list[str] = Field(min_length=MIN_EVIDENCE)
    confidence: float = Field(ge=0, le=1)


@dataclass(frozen=True)
class Checked:
    draft: Draft | None  # None when rejected
    problems: list[str]  # why it was rejected; empty when it passed


def check(raw: dict[str, Any], facts: list[Fact]) -> Checked:
    try:
        draft = Draft.model_validate(raw)
    except ValidationError as exc:
        problems = [f"{'.'.join(map(str, e['loc'])) or 'answer'}: {e['msg']}" for e in exc.errors()]
        return Checked(None, problems)

    by_id = {fact.id: fact for fact in facts}
    problems = [
        f"evidence cites {fact_id}, which does not exist"
        for fact_id in draft.evidence
        if fact_id not in by_id
    ]
    for field_name in ("summary", "hypothesis"):
        for sentence in SENTENCE_END.split(getattr(draft, field_name)):
            problems += _sentence_problems(field_name, sentence, by_id)
    return Checked(None if problems else draft, problems)


def citations(text: str) -> list[str]:
    """The fact IDs cited in a text, in order: "[F2] ... [F3, F5]" gives F2, F3, F5."""
    return [fact_id.strip() for group in CITATION.findall(text) for fact_id in group.split(",")]


def _sentence_problems(field_name: str, sentence: str, by_id: dict[str, Fact]) -> list[str]:
    cited = citations(sentence)
    missing = [fact_id for fact_id in cited if fact_id not in by_id]
    if missing:
        return [f"{field_name} cites {', '.join(missing)}, which does not exist"]
    numbers = [float(n) for n in NUMBER.findall(NAMED_NUMBER.sub("", DATE.sub("", sentence)))]
    if not numbers:
        return []
    if not cited:
        return [f"{field_name}: a sentence with numbers cites no fact: {sentence!r}"]
    allowed = {float(number) for fact_id in cited for number in by_id[fact_id].numbers}
    return [
        f"{field_name}: {_show(number)} is not in {', '.join(cited)}"
        for number in numbers
        if number not in allowed
    ]


def _show(number: float) -> str:
    return str(int(number)) if number.is_integer() else str(number)
