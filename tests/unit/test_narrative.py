"""The narrative's checks, with a fake LLM: the validator, the confidence caps, and the
retry-then-fallback flow. Eval case 2 (an invented number) is tested here, as it needs no LLM."""

from dataclasses import replace

import openai
import pytest
from test_metrics import PRS, WINDOW  # the example worked out by hand there

from capacity_radar.analytics.metrics import capacity_report
from capacity_radar.analytics.queries import DataQuality
from capacity_radar.narrative.confidence import capped_confidence
from capacity_radar.narrative.evidence import Fact
from capacity_radar.narrative.service import fallback, write_narrative
from capacity_radar.narrative.validator import check

FACTS = [
    Fact(
        "F1", "From 2026-07-29 to 2026-10-19, 383 PRs arrived; 233 got a first review", (383, 233)
    ),
    Fact(
        "F2",
        "PRs waiting for a first review: 205 at the start of the window, 234 at the end",
        (205, 234),
    ),
    Fact("F3", "197 of the 234 waiting PRs (84%) had waited more than 30 days", (197, 234, 84, 30)),
    Fact(
        "F4",
        "Week of 2026-09-14: median wait for a first review 90 h vs 38.2 h (z = 4.1)",
        (90, 38.2, 4.1),
    ),
]
GOOD = {
    "summary": "The queue grew from 205 to 234 PRs [F2]. "
    "Most of it is old: 84% waited over 30 days [F3].",
    "hypothesis": "Review capacity likely goes to new PRs, leaving older ones behind [F3].",
    "evidence": ["F2", "F3"],
    "confidence": 0.7,
}
INVENTED = {**GOOD, "summary": "The queue doubled to 410 PRs [F2]."}  # eval case 2

REPORT = capacity_report(PRS, WINDOW)  # 2 reviewed PRs, 2 full weeks: very little data
PLENTY = replace(REPORT, wait_sample_size=120, weeks=[replace(REPORT.weeks[0])] * 10)
QUALITY = DataQuality(
    prs_created=50, bot_prs=0, drafts_excluded=0, merged_prs=40, merged_with_review=36
)


class FakeLLM:
    """Gives the queued answers in turn, and records the feedback of each call."""

    model = "fake"

    def __init__(self, *answers):
        self.answers = list(answers)
        self.feedback = []

    async def draft(self, facts, feedback=None):
        self.feedback.append(feedback)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


# --- validator ---


def test_a_grounded_draft_passes():
    assert check(GOOD, FACTS).problems == []


def test_an_invented_number_is_rejected():
    assert check(INVENTED, FACTS).problems == ["summary: 410 is not in F2"]


def test_a_number_must_come_from_the_fact_cited_next_to_it():
    draft = {**GOOD, "summary": "The queue ended at 234, and 383 PRs arrived [F2]."}  # 383 is F1's
    assert check(draft, FACTS).problems == ["summary: 383 is not in F2"]


def test_a_sentence_with_numbers_needs_a_citation():
    draft = {**GOOD, "hypothesis": "Most of the 234 waiting PRs are old."}
    assert check(draft, FACTS).problems == [
        "hypothesis: a sentence with numbers cites no fact: 'Most of the 234 waiting PRs are old.'"
    ]


def test_several_facts_can_be_cited_together():
    # A real gpt-4.1-mini answer: one sentence, numbers from two facts, cited as [F2, F3].
    draft = {
        **GOOD,
        "summary": "The queue grew from 205 to 234, and 84% waited over 30 days [F2, F3].",
    }
    assert check(draft, FACTS).problems == []


def test_citations_must_exist():
    draft = {**GOOD, "evidence": ["F2", "F9"], "hypothesis": "Capacity is thin [F8]."}
    assert check(draft, FACTS).problems == [
        "evidence cites F9, which does not exist",
        "hypothesis cites F8, which does not exist",
    ]


def test_dates_and_percentile_names_are_not_numbers_to_check():
    draft = {**GOOD, "summary": "In the week of 2026-09-14, the median wait was 90 h [F4]."}
    assert check(draft, FACTS).problems == []


@pytest.mark.parametrize(
    "change",
    [{"confidence": 1.5}, {"evidence": ["F2"]}, {"summary": ""}, {"hypothesis": None}],
)
def test_the_shape_is_checked(change):
    assert check({**GOOD, **change}, FACTS).draft is None


# --- confidence caps ---


def test_little_data_caps_the_confidence():
    confidence = capped_confidence(0.8, REPORT, QUALITY)
    assert confidence.value == 0.5
    assert confidence.notes == [
        "capped at 0.5: only 2 PRs got a first review in the window",
        "capped at 0.6: only 2 full weeks of data",
    ]


def test_enough_data_keeps_the_llm_confidence():
    confidence = capped_confidence(0.8, PLENTY, QUALITY)
    assert (confidence.value, confidence.notes) == (0.8, [])


def test_reviews_outside_github_cap_the_confidence():  # eval case 5's code-side check
    few_reviews = replace(QUALITY, merged_with_review=5)  # 5 of 40 merged PRs
    assert capped_confidence(0.8, PLENTY, few_reviews).value == 0.5


def test_a_low_confidence_is_never_raised():
    assert capped_confidence(0.2, REPORT, QUALITY).value == 0.2


# --- retry and fallback ---


async def test_a_good_first_draft_is_used():
    narrative = await write_narrative(FakeLLM(GOOD), FACTS, PLENTY, QUALITY)
    assert (narrative.source, narrative.confidence, narrative.evidence) == (
        "llm",
        0.7,
        ["F2", "F3"],
    )


async def test_a_rejected_draft_is_retried_with_its_problems():
    llm = FakeLLM(INVENTED, GOOD)
    narrative = await write_narrative(llm, FACTS, PLENTY, QUALITY)
    assert narrative.source == "llm"
    assert "summary: 410 is not in F2" in llm.feedback[1]


async def test_an_invented_number_never_reaches_the_response():  # eval case 2
    narrative = await write_narrative(FakeLLM(INVENTED, INVENTED), FACTS, PLENTY, QUALITY)
    assert (narrative.source, narrative.confidence) == ("fallback", 0.3)
    assert "410" not in narrative.summary + narrative.hypothesis
    assert narrative.problems == ["summary: 410 is not in F2"]


async def test_an_llm_outage_falls_back_without_retrying():
    outage = openai.APIConnectionError(request=None)
    llm = FakeLLM(outage, GOOD)
    narrative = await write_narrative(llm, FACTS, PLENTY, QUALITY)
    assert narrative.source == "fallback" and len(llm.feedback) == 1


async def test_a_reply_that_is_not_json_is_retried():
    narrative = await write_narrative(FakeLLM(ValueError(), GOOD), FACTS, PLENTY, QUALITY)
    assert narrative.source == "llm"


def test_the_fallback_passes_its_own_checks():
    narrative = fallback(FACTS, ["anything"])
    draft = {
        "summary": narrative.summary,
        "hypothesis": narrative.hypothesis,
        "evidence": narrative.evidence,
        "confidence": narrative.confidence,
    }
    assert check(draft, FACTS).problems == []
