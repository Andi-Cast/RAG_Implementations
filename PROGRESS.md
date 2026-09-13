# Project Progress

Tracking against the build order from the project brief. Checked off as we complete things.

## 1. Repo scaffold + environment + vector store
- [x] `uv` environment (`pyproject.toml`, `.venv`, `uv.lock`)
- [x] `docker-compose.yml` — Postgres + pgvector running, `vector` extension confirmed
- [x] `.gitignore` (incl. `.venv/`, `.claude/`, `data/raw/`, etc.)

## 2. Data pipeline
- [x] Synthea generated locally (200 patients, `exporter.text.per_encounter_export`) — 16,343 per-encounter notes
- [x] Chunker (`src/rag/ingest/chunker.py`) — `split_top_level_sections`, `split_encounter_subsections`, `chunk_encounter_file`
- [x] `tests/test_chunker.py` (+ `tests/fixtures/sample_encounter.txt`)
- [x] Clearance-tier tagging (`src/rag/ingest/clearance_tagger.py`) — section default + fail-closed fallback + keyword escalation, + tests
- [x] Shared `src/rag/models.py` (`Chunk`, `CLEARANCE_TIERS`)
- [x] PII span labeling (`src/rag/ingest/pii_labeler.py`) — ground-truth spans (name, birth_date) + tests

## 3. Eval harness + gold set (built first, ahead of any retrieval code)
- [x] Retrieval metrics (`src/rag/eval/retrieval_metrics.py`) — `recall_at_k`, `reciprocal_rank`, `ndcg_at_k` + tests
- [x] Security metrics (`src/rag/eval/security_metrics.py`) — `access_control_leakage`, `injection_defense_success_rate`, `pii_redaction_recall` + tests
- [ ] Generation metrics (RAGAS/DeepEval — faithfulness, answer relevance)
- [~] Gold set (`data/gold/gold_set.json`) — 14-query pilot set drafted, spot-checked but not yet hand-verified as fully frozen; expand toward 50-100 later
- [x] Harness orchestration (`src/rag/eval/harness.py`) — `load_gold_set`, `run_retrieval_eval` (pluggable `retrieve_fn`)

## 4. Retrieval ladder (each rung benchmarked against the frozen gold set)
- [x] Naive dense top-k (baseline) — `src/rag/retrieval/dense.py`; corpus loaded (91,969 chunks, `bge-small-en-v1.5`, pgvector); benchmarked: recall@10=0.536, MRR=0.381, nDCG@10=0.413
- [x] + Hybrid (BM25-style + RRF fusion) — `sparse_bm25.py` (Postgres full-text search, AND→OR converted), `fusion_rrf.py`, `hybrid.py` (candidate_k=50 widened pool before fusion); benchmarked: recall@10=0.607, MRR=0.534, nDCG@10=0.532
- [ ] + Cross-encoder reranking
- [ ] + Contextual compression

## 5. Security layer + evals
- [ ] RBAC pre-filter (metadata filter *inside* the ANN query, never post-filter)
- [ ] Prompt-injection defense (hand-rolled)
- [x] PII detection (hand-rolled) — `src/rag/security/pii_redaction.py` (regex structured PII + naive name heuristic, known false-neg/false-pos documented in tests) + tests
- [ ] Actual redaction/masking step (currently only detects spans, doesn't mask text yet)

## 6. Bedrock integration
- [ ] Generation-model comparison axis
- [ ] Guardrails (PII + injection) benchmarked vs. hand-rolled
- [ ] Knowledge Bases — one managed-RAG baseline row

## 7. FastAPI serving
- [ ] API with role-based filtering injected from authenticated identity

## 8. Writeup
- [ ] README with benchmark tables (retrieval × generation model × managed-vs-custom security)
- [ ] "Techniques deliberately not used, and why" section

## Known limitations / improvement ideas (revisit once v1 works end-to-end)
- `naive_dense_retrieve` opens and closes a brand-new DB connection on every call — fine for today's 14-query pilot, wasteful once benchmarking runs many queries in a row. Should accept a reusable connection instead.
- The embedding model name (`"BAAI/bge-small-en-v1.5"`) is a duplicated magic string in both `scripts/load_corpus.py` and `src/rag/retrieval/dense.py`. Same class of bug we already hit once with `Chunk`/`CLEARANCE_TIERS` — if the model ever changes, both places need updating in sync or the corpus and query embeddings stop being comparable. Should become one shared constant.
- Clearance tagger's keyword escalation has a known false-positive: "Abuse-Deterrent" (a real opioid formulation term) triggers the "abuse" keyword and gets tagged `restricted` for the wrong reason — outcome is arguably still fine here (opioid chunks are genuinely sensitive) but it's the wrong reason, and could misfire elsewhere.
- Naive dense retrieval observation: patients with many repeat encounters produce several near-duplicate chunks (e.g. the same medication mentioned across multiple visits) that compete for top-k slots — a query can retrieve the *right patient's* correct-content chunk from the *wrong encounter* rather than the exact `chunk_id` labeled in the gold set. Worth watching once real Recall@k/nDCG numbers come in — may indicate the gold set needs less ambiguous single-answer queries, or that this is a genuine, expected retrieval limitation worth discussing in the writeup.
- PII detector's known false-negative (digit-suffixed Synthea names) and false-positive (all-caps facility names) are already documented as tests in `test_pii_redaction.py`, not just narrative — flagging here too for visibility when writing the final report.
- `sparse_retrieve` builds the same OR-converted tsquery twice in one SQL statement (once for `WHERE`, once for `ORDER BY`) — redundant computation, not incorrect. Could be computed once in a CTE/subquery instead.
- `sparse_retrieve` required converting `plainto_tsquery`'s default AND-joined terms into an OR-joined query (via `regexp_replace`) — Postgres full-text search's default AND semantics are a hard filter (a chunk missing even one query term is excluded entirely), unlike real BM25's soft/weighted scoring where partial term matches still rank, just lower. Worth naming explicitly in the writeup as a deliberate deviation from "true" BM25, consistent with the project's earlier honesty about `ts_rank` not being the literal BM25 formula.
