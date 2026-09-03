# Secure Multi-Technique RAG

A Retrieval-Augmented Generation system built over a synthetic, access-controlled medical corpus — designed around a question most RAG portfolios skip: **what happens when different users are legally allowed to see different slices of the same data?**

The project has two goals, deliberately paired: (1) benchmark a ladder of retrieval techniques (naive dense → hybrid → reranked → compressed) against a frozen, hand-verified gold set, and (2) build a real security layer — role-based access control enforced *inside* the retrieval query, hand-rolled PII detection, and prompt-injection defense — that a production healthcare RAG system would actually need before it could ship.

**Status: in progress.** See [`PROGRESS.md`](PROGRESS.md) for the live build checklist. This README describes what's actually built and tested today, not the finished system.

## Why this domain

The corpus is synthetic medical records (generated with [Synthea](https://github.com/synthetichealth/synthea)) rather than a generic document set, because HIPAA's "minimum necessary" rule gives access control a genuine reason to exist here: a nurse, a biller, and a behavioral-health specialist are supposed to see different parts of the same patient's chart. That constraint shapes the whole security design — access control is enforced as a metadata pre-filter *inside* the vector query, never a post-filter, because post-filtering would let restricted content pass through retrieval, logging, and reranking before being dropped, silently leaking its existence.

## What's built and tested so far

**Data pipeline** (`src/rag/ingest/`)
- Structure-aware chunker that parses Synthea's per-encounter clinical notes into section-level chunks (`chunker.py`)
- Clearance-tier tagging: a section-based default with a **fail-closed** fallback (an unrecognized section defaults to the *most* restrictive tier, not the least) plus keyword-based escalation for sensitive content that shows up regardless of section — e.g. an abuse-related finding filed under a routine `CONDITIONS` section still gets escalated (`clearance_tagger.py`)
- Ground-truth PII span labeling, using Synthea's own known identifier values rather than guessing (`pii_labeler.py`)

**Security** (`src/rag/security/`)
- A hand-rolled PII detector combining regex for structured PII (dates, SSNs, phone numbers) with a naive Title-Case heuristic for names — deliberately kept naive to demonstrate *why* regex-only PII detection fails on unstructured text: it misses Synthea's own digit-suffixed synthetic names entirely, while false-positively flagging facility names. Both failure modes are captured as explicit regression tests, not swept under the rug (`pii_redaction.py`)

**Evaluation harness** (`src/rag/eval/`) — built *before* any retrieval technique, so every technique that follows is measured on identical ground:
- Retrieval metrics: Recall@k, MRR, nDCG@k, implemented from the formulas rather than a library, each with edge cases (empty ground truth, `k` beyond the result count) explicitly handled
- Security metrics: access-control leakage rate, injection-defense success rate, PII-redaction recall
- A pluggable harness (`harness.py`) that scores any retrieval function against the gold set — the same harness runs unchanged for every rung of the retrieval ladder, you just swap the function passed in

**Gold set** (`data/gold/gold_set.json`) — 14 hand-drafted queries against real corpus chunks, spanning both routine clinical facts and `restricted`-tier content (substance-use screening, an intimate-partner-abuse finding), including one multi-chunk query. This is a *pilot* set for getting the pipeline working end-to-end; the plan is to expand toward 50-100 verified queries before final benchmark numbers are reported.

**Retrieval** (`src/rag/retrieval/`)
- Naive dense retrieval — the first rung of the ladder: embeds the query with the same model used to embed the corpus (`BAAI/bge-small-en-v1.5`), then ranks chunks by cosine distance via pgvector's `<=>` operator directly in SQL (`dense.py`)

**Infrastructure**: Postgres + pgvector (metadata filtering inside the ANN query is the whole reason this vector store was chosen over alternatives that don't support it well), the full corpus embedded and loaded (91,969 chunks across 200 synthetic patients), `uv`-managed environment, `pytest` suite covering every module above.

## Preliminary results

Naive dense retrieval scored against the 14-query pilot gold set:

| Technique | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|
| Naive dense (baseline) | 0.536 | 0.381 | 0.413 |

**Read these as early signal, not final numbers** — the gold set is a 14-query pilot (target is 50-100 hand-verified queries before this table is treated as authoritative), and this is only the first of four ladder rungs. One concrete, expected failure mode already observed: patients with many repeat encounters produce several near-duplicate chunks (the same medication mentioned across multiple visits), so a query can retrieve the *right patient's* correct content from the *wrong specific encounter* rather than the exact chunk labeled in the gold set — a plausible source of some of the recall misses here, and something the hybrid/reranking rungs may or may not improve on.

## What's next

- Hybrid (BM25 + RRF fusion), cross-encoder reranking, contextual compression — rungs 2-4 of the retrieval ladder
- The RBAC pre-filter, wired into the retrieval query itself
- Prompt-injection defense
- AWS Bedrock integration: a generation-model comparison axis, Guardrails benchmarked against the hand-rolled security layer, and a Bedrock Knowledge Base as one managed-RAG baseline row
- FastAPI serving with role derived from authenticated identity, never from client input

## Deliberately not used (and why this matters for evaluating the project)

This project intentionally does not reach for LangChain, agentic RAG, GraphRAG, or Self-RAG — the orchestration is hand-rolled specifically so retrieval, fusion, and the security layer stay inspectable rather than hidden inside a framework abstraction. A full "techniques considered and rejected" writeup will land in the final README once the retrieval ladder is complete.

## Running it

```bash
uv sync
docker compose up -d
docker compose exec -T postgres psql -U rag -d rag -f - < src/rag/db/schema.sql
uv run pytest tests/
```
