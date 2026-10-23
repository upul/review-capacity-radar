# Notes
This document outlines how to run the service, explains how the metrics are calculated, provides an overview of the system architecture, and discusses future directions and the use of AI in the development process.

## 1. How to run it

Prerequisites: Docker; a fine-grained GitHub token with read-only access to public
repositories; an OpenAI.

```bash
cp .env.example .env     # set GITHUB_TOKEN, LLM_API_KEY, LLM_MODEL (e.g. gpt-4.1-mini)
docker compose up --build -d
docker compose logs -f app   # first sync of scikit-learn: a few minutes
curl -s localhost:8000/api/v1/repos/scikit-learn/scikit-learn/insights/review-capacity | jq .
curl -s localhost:8000/api/v1/repos/scikit-learn/scikit-learn/insights/queue-forecast | jq .forecast
curl -s localhost:8000/api/v1/repos/scikit-learn/scikit-learn/insights/narrative | jq .
```

## 2. The metric: what it means and why

Leaderboards ("top reviewers") reward volume but don't say whether a project keeps up with
its contributors. A queue does. For each pull request:

```
created ── waiting ──► first qualifying review          served
        └─ waiting ──► merged or closed with no review  exited unreviewed
        └─ waiting ──► still waiting at the window end   queue
```

- **Eligible PR:** not opened by a bot, not a draft. **Qualifying review:** submitted, not by
  the PR's author, not by a bot. (On GitHub, an author's replies in review threads are
  stored as reviews by the author; counting them would make waits look far shorter. A
  real-PR test checks this against GitHub data.)
- Over a date window: **arrivals**, **served**, **exited unreviewed**, the **queue** at start
  and end (with how much of it is stale, over 30 days), and the **wait** to first review
  (median and p90, because the tail is long).
- **Flow balance:** queue at start + arrivals − served − exited = queue at end. Each term is
  counted independently, so it must always hold; the response includes it so anyone can
  check the numbers, and a randomized test checks it over 200 random histories.
- **Little's Law** (L ≈ λ × W) relates queue size, arrival rate and time in queue, all
  measured separately. A large gap means the queue isn't steady; on scikit-learn it is 34%,
  because old PRs pile up and never leave.
- **Forecast:** a Monte Carlo replay of the last 12 complete weeks (2,000 futures, fixed
  seed), giving the queue's p10 / p50 / p90 per week ahead.
- **Unusual weeks:** a leave-one-out z-score (each week against the mean and SD of the other
  weeks), flagged above 3. Leaving the week out matters: with 11 weeks, a plain z-score can
  never exceed 3.02, however extreme the week.

On scikit-learn (29 Jul to 19 Oct 2026): 383 PRs arrived, 233 got a first review, 121 left
without one, and the queue grew from 205 to 234. New PRs are reviewed quickly (median 18.4
h), but 84% of the waiting PRs are over 30 days old: the backlog is a long tail that is
rarely picked up, more than a shortage of reviewers.

## 3. Architectural Decisions
Main decisions:

- **Persist, then serve.** Insight requests cost no GitHub calls and stay fast; the sync is
  incremental, so after the first run a refresh takes a few requests.
- **Maths in Python, SQL for facts.** Every metric is testable without a database and
  checked by hand-worked examples, the flow-balance invariant, and real GitHub data.
- **Evidence chain enforced by code, not by the prompt.** The prompt asks for citations;
  the validator guarantees them.
- **Production habits:** settings validated at startup with secrets as `SecretStr`; every
  error is RFC 7807 problem details; a health check with a timeout; no database connection
  held during the LLM call; a two-stage, non-root Docker image with no secrets in it;
  ruff, pre-commit and secret scanning on every commit.

## 4. Trade-offs and what I chose not to do

- **No migrations:** tables are created at startup (`create_all`). Fine for a fresh
  deployment; schema changes on existing data need Alembic.
- **One background loop, one instance.** Syncs can't overlap, so there is no lock. Running
  several instances would need a Postgres advisory lock per repository, or a separate worker.
- **Reviews are fetched one PR at a time.** Simple and gentle on the rate limit; the first
  sync is slower than it needs to be.
- **Draft status is the current one.** A PR that was a draft for a week counts as waiting
  from creation. GitHub's timeline events would give the exact ready-for-review time.
- **Only GitHub reviews count.** Projects that review in comments or elsewhere look
  unreviewed; `merged_with_review_share` reports this, and the narrative's confidence is
  capped when it is below 50%.
- **`capacity_limited` is weak on repos with a stale backlog:** the queue never empties, so
  every week qualifies. It should look at the fresh queue only (next step).
- **The forecast assumes the next weeks resemble the last 12,** and replays whole weeks, so
  with 11 weeks of history the percentiles move in steps.
- **The LLM's reasoning is not validated, only its numbers.** gpt-4.1-mini tends to read a
  growing queue as "not enough capacity" even when the facts point to a long tail. The
  validator stops invented numbers; evals are what would measure reasoning (next step).
- **No API authentication or rate limiting:** it serves public data locally. The narrative
  endpoint calls a paid API, so a shared deployment would need both.

## 5. Next steps with another day

1. **Eval harness for the narrative**, run before any prompt or model change. The cases are
   designed: a clear backlog (must cite the queue and stale facts), a stable repo (must not
   invent a problem), a tiny repo (confidence capped), reviews outside GitHub (must flag
   incomplete data), and an invented number (must fall back; already a unit test).
2. **Alembic** migrations.
3. **A small UI** the weekly series as a chart, the forecast fan, and the narrative with
   its cited facts.
4. **More comprehensive unit test suite** to cover all the calculations we do in the backend.

## 6. How I used AI

I used Claude (Anthropic's assistant) throughout, as a pair programmer and tutor.

- **Planning:** brainstorming project ideas against the brief, and the design documents and
  diagrams. It first drifted from the brief; I made it re-read the requirements, which led
  to this project.
- **Code:** it drafted most of the code and tests, one file at a time. I reviewed each
  change, ran it locally against real data, and made every commit myself.
- **Learning:** I asked for hand-worked examples before accepting any method: the flow
  balance, the forecast's percentiles, the z-score and Little's Law. Several of those
  examples are now the unit tests.
- **Decisions I made or changed:** the scope (REST rather than GraphQL, Postgres without
  Alembic, scikit-learn as the subject), building step by step instead of in large batches,
  the real-PR fixtures I picked and checked against GitHub, choosing a leave-one-out z-score
  over the MAD-based score it first proposed, the narrative prompt and its eval cases, and
  cutting a planned sync API down to the background loop after re-reading the brief.
