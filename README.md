# Secure Multi-Technique RAG

A Retrieval-Augmented Generation system built over a synthetic, access-controlled medical corpus — designed around a question most RAG portfolios skip: **what happens when different users are legally allowed to see different slices of the same data?**

The project has two goals, deliberately paired: (1) benchmark a ladder of retrieval techniques (naive dense → hybrid → reranked → compressed) against a frozen, hand-verified gold set, and (2) build a real security layer — role-based access control enforced *inside* the retrieval query, hand-rolled PII detection, and prompt-injection defense — that a production healthcare RAG system would actually need before it could ship.

**Status: core system complete.** The full retrieval ladder, the entire security layer, and a generation-model comparison are all built and benchmarked end-to-end against a real corpus with real queries — see the results below. Bedrock Guardrails/Knowledge Base comparisons, FastAPI serving, and a larger gold set are planned next (see [`PROGRESS.md`](PROGRESS.md) for the live build checklist).

## Why this domain

The corpus is synthetic medical records (generated with [Synthea](https://github.com/synthetichealth/synthea)) rather than a generic document set, because HIPAA's "minimum necessary" rule gives access control a genuine reason to exist here: a nurse, a biller, and a behavioral-health specialist are supposed to see different parts of the same patient's chart. That constraint shapes the whole security design — access control is enforced as a metadata pre-filter *inside* the vector query, never a post-filter, because post-filtering would let restricted content pass through retrieval, logging, and reranking before being dropped, silently leaking its existence.

## Architecture

```mermaid
graph TD
    subgraph Ingestion
        A[Synthea clinical notes] --> B[chunker.py]
        B --> C[clearance_tagger.py]
        C --> D[embed: bge-small-en-v1.5]
    end

    D --> E[(Postgres + pgvector)]

    subgraph "Retrieval ladder — RBAC pre-filter applied inside every SQL query"
        E --> F[naive_dense_retrieve]
        E --> G[sparse_retrieve]
        F --> H[hybrid_retrieve: RRF fusion]
        G --> H
        H --> I[cross_encoder_rerank]
        I --> J[compress_chunk: line-level extraction]
    end

    J --> K[sanitize_retrieved_context: injection defense]
    K --> L[generate_answer: Bedrock Converse API]
    L --> M{Claude Haiku 4.5 / gpt-oss-120b}
    M --> N[redact_pii: mask output]

    subgraph "Eval harness — scores every stage above"
        O[gold_set.json] --> P[run_retrieval_eval]
        O --> Q[run_generation_eval]
        P --> R["Recall@k / MRR / nDCG@k"]
        Q --> S["faithfulness / answer_relevance / answer_correctness"]
    end
```

## What's built and tested so far

**Data pipeline** (`src/rag/ingest/`)
- Structure-aware chunker that parses Synthea's per-encounter clinical notes into section-level chunks (`chunker.py`)
- Clearance-tier tagging: a section-based default with a **fail-closed** fallback (an unrecognized section defaults to the *most* restrictive tier, not the least) plus keyword-based escalation for sensitive content that shows up regardless of section — e.g. an abuse-related finding filed under a routine `CONDITIONS` section still gets escalated (`clearance_tagger.py`)
- Ground-truth PII span labeling, using Synthea's own known identifier values rather than guessing (`pii_labeler.py`)

**Security** (`src/rag/security/`)
- A hand-rolled PII detector combining regex for structured PII (dates, SSNs, phone numbers) with a naive Title-Case heuristic for names — deliberately kept naive to demonstrate *why* regex-only PII detection fails on unstructured text: it misses Synthea's own digit-suffixed synthetic names entirely, while false-positively flagging facility names. Both failure modes are captured as explicit regression tests, not swept under the rug (`pii_redaction.py`)
- RBAC pre-filter: `allowed_tiers()` computes every clearance tier at or below a user's own rank, which every retrieval function's SQL query then filters on directly (`WHERE clearance_tier = ANY(%s)`) *before* ranking happens — never a post-filter. `user_clearance` is a required parameter with no default on every retrieval function, deliberately: a security-critical parameter defaulting to "see everything" would be a fail-*open* bug waiting to happen (`access_filter.py`)
- Person-level pre-filter: the realistic clinical workflow is a clinician with *one specific patient's* full chart open, not a search across every patient in the system — so every retrieval function also takes a required (no default) `person_id` parameter, filtered inside the same SQL `WHERE` clause alongside `clearance_tier`, `None` only to deliberately opt into cross-person/population-level search. Finding this mattered came from a real bug: the chunks table's own `patient_id` column turned out to be *encounter*-scoped, not person-scoped (Synthea exports one file per visit and the chunker used the filename as-is — see `scripts/backfill_person_id.py` for the fix and `PROGRESS.md` for the full story), which the benchmark numbers below expose directly

  ```mermaid
  graph LR
      A[Query + user_clearance + person_id] --> B["WHERE clearance_tier = ANY(allowed_tiers)\nAND person_id = %s\nORDER BY embedding &lt;=&gt; query"]
      B --> C[Only this person's allowed chunks are ever ranked, logged, or returned]
  ```

  ```mermaid
  graph LR
      subgraph "This project: pre-filter"
          A1[Query + user_clearance] --> B1["WHERE clearance_tier = ANY(allowed_tiers)\nORDER BY embedding <=> query"]
          B1 --> C1[Only allowed chunks are ever ranked, logged, or returned]
      end

      subgraph "Common mistake: post-filter"
          A2[Query] --> B2["ORDER BY embedding <=> query\n(no clearance filter)"]
          B2 --> C2[Top-k results, restricted chunks included]
          C2 --> D2[Drop restricted chunks after the fact]
          D2 --> E2["Restricted content already touched ranking,\nlogs, and the reranker before being dropped"]
      end
  ```
- Prompt-injection defense: a hand-rolled detector (`detect_injection_patterns`) flags instruction-override phrasing ("ignore previous instructions"), role-play override attempts ("you are now..."), and fake system/assistant turn markers planted inside retrieved text, plus a sanitizer (`sanitize_retrieved_context`) that redacts matched spans before they reach the prompt. Deliberately naive, same tradeoff as the PII detector — a documented test shows a simple rephrasing evades it entirely (`injection_defense.py`)
- PII redaction: `redact_pii` turns `detect_pii`'s spans into actually-masked output text, replacing right-to-left (sorted by start index, descending) so masking one span never corrupts the character indices of spans still waiting to be replaced (`pii_redaction.py`)

**Evaluation harness** (`src/rag/eval/`) — built *before* any retrieval technique, so every technique that follows is measured on identical ground:
- Retrieval metrics: Recall@k, MRR, nDCG@k, implemented from the formulas rather than a library, each with edge cases (empty ground truth, `k` beyond the result count) explicitly handled
- Security metrics: access-control leakage rate, injection-defense success rate, PII-redaction recall
- A pluggable harness (`harness.py`) that scores any retrieval function against the gold set — the same harness runs unchanged for every rung of the retrieval ladder, you just swap the function passed in

**Gold set** (`data/gold/gold_set.json`) — 250 queries against real corpus chunks: the original 14 hand-drafted (spanning routine clinical facts and `restricted`-tier content like substance-use screening and an intimate-partner-abuse finding, including one multi-chunk query), plus 236 generated deterministically by `scripts/generate_gold_set_candidates.py` — a template-based parser, **not an LLM**, that reads each sampled chunk's own structured `date : description (type)` text directly and builds a question/answer pair from it, so every answer is correct by construction rather than generated-then-hoped-to-be-right. Stratified ~70/30 internal/restricted (preserving the original set's deliberate restricted-tier emphasis against the real corpus's ~93/7 split) across 9 sections, excluding pure metadata sections (`DEMOGRAPHICS`, and `ENCOUNTER_META` for `internal`).

**Retrieval** (`src/rag/retrieval/`)
- Naive dense retrieval — rung 1: embeds the query with the same model used to embed the corpus (`BAAI/bge-small-en-v1.5`), then ranks chunks by cosine distance via pgvector's `<=>` operator directly in SQL (`dense.py`)
- Hybrid retrieval — rung 2: combines dense retrieval with keyword-based search via Postgres full-text search (`sparse_bm25.py`), merged with Reciprocal Rank Fusion (`fusion_rrf.py`). Each sub-retriever pulls a wider candidate pool (50) before fusion narrows to the final top-k, so a chunk ranked outside one method's top-10 can still surface if the other method ranks it highly (`hybrid.py`)
- Cross-encoder reranking — rung 3: takes hybrid's top-50 candidate pool and scores each `(query, chunk_text)` pair directly with `cross-encoder/ms-marco-MiniLM-L-6-v2`, a slower but more accurate model than the bi-encoder used for dense retrieval, since it reasons about the query and candidate jointly rather than comparing precomputed vectors (`rerank.py`)
- Contextual compression — rung 4: extracts (not summarizes) the most query-relevant lines out of an already-retrieved chunk by embedding each line and comparing it to the query, the same bi-encoder similarity math as dense retrieval, just applied at line-level instead of chunk-level. Doesn't change which chunks get retrieved, so it isn't a new row in the retrieval benchmark table below — its payoff is shorter, cleaner context for the eventual generation step (`compression.py`)

**Infrastructure**: Postgres + pgvector (metadata filtering inside the ANN query is the whole reason this vector store was chosen over alternatives that don't support it well), the full corpus embedded and loaded (91,969 chunks across 234 synthetic people — not 200; Synthea's population generator runs extra attempts to reach a target of 200 *living* patients and leaves the rest behind as completed records), `uv`-managed environment, `pytest` suite covering every module above.

## Results

### Retrieval ladder, person-scoped (the realistic scenario)

This is the headline result, and it only exists because of a real bug found and fixed mid-project (see "A real bug, found honestly" below). Scored against the full 250-query gold set, retrieval scoped to the one real person each query is actually about — the realistic clinical workflow, where a clinician already has *this specific patient's* chart open rather than searching the whole system:

| Technique | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|
| Naive dense (baseline) | 0.430 | 0.260 | 0.300 |
| Hybrid (dense + keyword, RRF) | 0.586 | 0.432 | 0.468 |
| + Cross-encoder rerank | **0.710** | **0.588** | **0.617** |

![Retrieval ladder benchmark chart, person-scoped](docs/images/person_scoped_ladder.png)

Every single rung improves on every single metric — the cleanest, most textbook-shaped result this project has produced. Naive dense → hybrid is +36% relative recall; hybrid → rerank is another +21%; naive dense → rerank overall is +65%. Unlike the unscoped numbers below, nothing here hits a ceiling or moves backwards, because this is the problem the retrieval ladder is actually meant to solve: finding the right fact inside one person's real, multi-visit chart, not finding the right person out of 234.

Also worth naming honestly: "hybrid" here uses Postgres full-text search (`ts_rank`) rather than the literal BM25 formula, and required converting its default AND-matching into OR-matching to behave like a real ranked retriever rather than an all-or-nothing filter — a deliberate, documented deviation, not an oversight (see `PROGRESS.md`'s improvement log for details).

Contextual compression (rung 4) is built and verified but deliberately doesn't add a fifth row here — it doesn't change which chunks are retrieved, only trims their content, so its actual payoff (shorter context for generation) isn't visible until generation metrics exist.

### A real bug, found honestly: "patient-scoped" retrieval was actually encounter-scoped

Patient-scoping was added to this project specifically because unscoped cross-person search is a weird benchmark for a system whose own motivating scenario is "a clinician looking at one patient's chart" — searching everyone's records for every query isn't what that workflow needs. The first implementation added a `patient_id` filter using the chunks table's existing `patient_id` column — and the very first benchmark run came back with **recall@10 = 1.0 across every technique**, which was suspicious enough to investigate rather than celebrate.

The cause: `patient_id` is `Path(file_path).stem` from `chunker.py` — and Synthea's `per_encounter_export` writes **one file per visit**, not one per person. So `patient_id` was actually an encounter identifier (`Georgie506_Batz141_7a327049-...-a1`, `...-a10`, `...-a11`, each a different visit of the same real person), and "patient-scoped" retrieval was really "scope to this one specific office visit's 3-10 chunks" — a search space small enough to make recall trivially perfect, independent of whether any retrieval technique was good.

The fix, without re-chunking or re-embedding anything: a new `person_id` column, backfilled (`scripts/backfill_person_id.py`) by matching the standard UUID shape Synthea embeds in every filename and dropping the encounter-index digits appended after it. Re-running the benchmark against the real person-level scope produced the actually-interesting, non-degenerate numbers in the table above. This is also how the "234 people, not 200" correction elsewhere in this README was discovered — independently cross-checked by a second extraction method before trusting it.

### Retrieval ladder, unscoped (population-level/research search)

The other legitimate use case — cross-person search across the whole corpus, e.g. "find patients with condition X" for research or quality reporting — scored against the same 250-query gold set with no person filter at all:

| Technique | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|
| Naive dense (baseline) | 0.102 | 0.068 | 0.076 |
| Hybrid (dense + keyword, RRF) | **0.330** | 0.177 | 0.213 |
| + Cross-encoder rerank | 0.306 | **0.193** | **0.219** |

![Retrieval ladder benchmark chart, unscoped](docs/images/unscoped_ladder.png)

These numbers are much worse, and much less clean (reranking's recall actually ticks down slightly from hybrid, 0.330 → 0.306), because this is a fundamentally harder problem: finding the right fact among 91,969 chunks across 234 people, many sharing near-identical recurring screenings and structurally templated questions ("what procedure was performed on `{date}`?"). Hybrid's relative improvement over naive dense is enormous here too (+224%) — for the same reason as the scoped table: Postgres full-text search matches a literal date token directly where dense embedding similarity can't (proven concretely in the worked example below). Read this table as "how hard is open-ended cross-patient search," not as "how good is this RAG system at its actual job" — that's what the table above is for.

### RBAC pre-filter results, person-scoped

Naive dense retrieval, same 250-query gold set, person-scoped, scored per role:

| Clearance | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|
| `restricted` (sees everything) | 0.438 | 0.262 | 0.303 |
| `internal` (can't see `restricted`-tier chunks) | **0.272** | **0.147** | **0.176** |

![RBAC pre-filter benchmark chart, person-scoped](docs/images/rbac_results.png)

`restricted`'s numbers are identical to the person-scoped baseline above — expected, since filtering to "everything at or below the top tier" is filtering to nothing at all. `internal`'s numbers are meaningfully lower across all three metrics, because the gold-set queries targeting `restricted`-tier content become unanswerable for that role — not because retrieval got worse, but because those chunks are now correctly invisible before ranking ever happens. That's the pre-filter design working exactly as intended, on the realistic per-person scenario rather than the harder cross-person one.

(The same comparison unscoped — `restricted`=0.102/0.068/0.076, `internal`=0.072/0.056/0.060 — shows the identical pattern at the harder scale; chart at `docs/images/rbac_results_unscoped.png`.)

### Generation-model comparison

**Note: this table still reflects the original 14-query pilot, unscoped, not yet re-run against the full 250-query gold set or the person-scoped retrieval above.** That re-run is a much larger, real-cost Bedrock operation (250 queries × 2 models × 3 metrics = 1,500 calls vs. this table's ~56), so it's deliberately not done automatically alongside the retrieval-ladder update above — it's a separate, explicit next step, and likely to show the same kind of improvement moving to person-scoped retrieval showed above.

Two Bedrock models (`generate_answer()`, Converse API) scored against the 14-query pilot gold set via hand-rolled LLM-as-judge metrics (`src/rag/eval/generation_metrics.py`) — no RAGAS/DeepEval dependency, same "implement from the concept" approach as the retrieval metrics. Retrieval context for this run came from `naive_dense_retrieve` (the baseline rung); Claude Haiku 4.5 acted as judge for both models.

| Model | Faithfulness | Answer Relevance | Answer Correctness |
|---|---|---|---|
| Claude Haiku 4.5 | 0.95 | 0.85 | 0.50 |
| gpt-oss-120b | 0.91 | 0.91 | 0.59 |

![Generation-model comparison benchmark chart](docs/images/generation_comparison.png)

Three distinct metrics, deliberately: **faithfulness** (is the answer grounded in the context it was actually given, i.e. no hallucination) and **answer relevance** (does the answer address the question asked) are both *reference-free* — they never look at the gold set's hand-written `reference_answer`. **Answer correctness** is the one metric that does, checking whether the generated answer actually conveys the same information as the known-correct answer.

That distinction is the whole story in this table: faithfulness and relevance both look strong for both models (~0.85-0.95), which on its own would read as "the system works well." Answer correctness tells a different, more honest story — both models land around 0.5-0.6, because `naive_dense_retrieve` itself fails to surface the right chunk on a meaningful fraction of queries (see the retrieval ladder's own recall numbers above). A model can be perfectly faithful to the *wrong* retrieved context and still be flatly incorrect — faithfulness/relevance alone would have hidden that failure completely; answer correctness is what surfaces it. Re-running this same generation benchmark against the better retrieval rungs (hybrid, cross-encoder rerank) rather than the naive baseline is the natural next measurement, since the retrieval ladder's own numbers suggest those rungs should close much of this gap.

One limitation worth naming directly: Haiku judges both models' answers, including its own — a single fixed judge keeps the comparison internally consistent, but a model judging its own output is a known potential source of self-favoring bias.

### Worked example: when good-looking metrics hide a real failure

A real trace from the gold set, correctly person-scoped (searching only this one real person's full chart, not a stranger's) — showing that even the realistic, easier version of this problem still has real failure cases, and exactly why three generation metrics exist instead of two:

> **Query:** "What condition was diagnosed for this patient on 2024-04-20?"
> **Gold-set reference answer:** "Acute viral pharyngitis"
> **Scoped to:** this patient's own real `person_id` (not a different person — the chart genuinely belongs to them, across all their visits)
>
> **Retrieved context** (`naive_dense_retrieve`, k=3) — the right person, the wrong visits:
> ```
> 2024-04-13 : Primary dental caries (disorder)
> 2024-04-13 : Gingival disease (disorder)
> 2022-02-19 : Viral sinusitis (disorder)
> 2025-01-07 : Acute bronchitis (disorder)
> ```
>
> **Generated answer** (Claude Haiku 4.5): *"Based on the provided medical records, no condition was diagnosed on 2024-04-20. The closest diagnoses to that date are from 2024-04-13... There is no record of any diagnosis on the specific date of 2024-04-20."*
>
> | Metric | Score | What it saw |
> |---|---|---|
> | Faithfulness | **1.0** | The answer is 100% consistent with the context it was shown — no hallucination |
> | Answer Relevance | **0.9** | The answer directly addresses the question asked |
> | Answer Correctness | **0.0** | The answer is simply wrong — the real diagnosis was pharyngitis |

Retrieval failed here even with the right person in scope: dense embeddings are good at topical similarity ("pharyngitis" ≈ "throat infection") but bad at exact structured matching — a date string like `2024-04-20` isn't semantically meaningful to a bi-encoder the way a clinical concept is, so among this one person's dozens of dated entries, the wrong visits still outranked the right one. The model then did the honest thing with what it was given. **Faithfulness and relevance both scored as if the system worked.** Only `answer_correctness` — the one metric that actually checks against ground truth — caught that it didn't. This is the concrete version of the aggregate ~57%-of-the-time gap in the person-scoped table above (naive dense's 0.430 recall@10), and the reason this project measures three generation metrics instead of two.

## What's next

- AWS Bedrock integration: Guardrails benchmarked against the hand-rolled security layer, and a Bedrock Knowledge Base as one managed-RAG baseline row
- FastAPI serving with role derived from authenticated identity, never from client input
- Expand the gold set from 14 to 50-100 hand-verified queries for more statistically meaningful benchmark numbers

## Deliberately not used (and why this matters for evaluating the project)

Every tradeoff below was a conscious scope decision, not an oversight — each one is also backed by a test or a benchmark number elsewhere in this README, not just asserted in prose.

- **No framework** (LangChain, agentic RAG, GraphRAG, Self-RAG). Orchestration is hand-rolled specifically so retrieval, fusion, and the security layer stay inspectable line-by-line rather than hidden inside a framework abstraction — the whole point of this project is to demonstrate understanding of *why* each technique works, not just call a library that does it.
- **No RAGAS/DeepEval for generation metrics.** `faithfulness`, `answer_relevance`, and `answer_correctness` are hand-rolled LLM-as-judge functions (`generation_metrics.py`), same "implement from the concept" approach as the retrieval metrics (`Recall@k`, `MRR`, `nDCG@k`). The tradeoff: a maintained library would have been faster to wire up, but the judge prompts and scoring logic here are fully inspectable rather than a black box.
- **Single ordinal clearance tier per user**, not role/attribute-based access control. `CLEARANCE_TIERS` models access as one linear scale (`public < internal < confidential < restricted`), where a real healthcare system often needs *non-nested* categories — a billing role and a behavioral-health role should see different, not strictly overlapping, slices of a chart, which an ordinal scale can't express. Kept deliberately for this project's scope (it's still a legitimate real-world pattern, e.g. classic security clearance levels), not a simplification I'm unaware of.
- **Full-text search (`ts_rank`) instead of literal BM25** for the hybrid retrieval rung, and its default AND-matching converted to OR-matching to behave like a ranked retriever rather than an all-or-nothing filter — a named, deliberate deviation (see `PROGRESS.md`'s improvement log), not a hidden approximation.
- **Regex/keyword-based PII detection and prompt-injection defense**, not an ML classifier or managed service. Both are deliberately naive, with their own false-negatives and false-positives captured as explicit regression tests rather than swept under the rug (digit-suffixed synthetic names evading the PII detector; a rephrased injection attempt evading the regex patterns). This is intentional groundwork for a planned comparison against Bedrock Guardrails — the hand-rolled baseline needs to exist, and needs its failure modes documented, before that comparison means anything.
- **One fixed judge model (Haiku) scoring both generation models**, including itself. Keeps the comparison internally consistent, but a model judging its own output is a known potential source of self-favoring bias — named directly rather than ignored.

## Running it

```bash
uv sync
docker compose up -d
docker compose exec -T postgres psql -U rag -d rag -f - < src/rag/db/schema.sql
uv run pytest tests/
```
