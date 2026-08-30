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
- [ ] PII span labeling (ground-truth spans for redaction scoring)

## 3. Eval harness + gold set (built first, ahead of any retrieval code)
- [x] Retrieval metrics (`src/rag/eval/retrieval_metrics.py`) — `recall_at_k`, `reciprocal_rank`, `ndcg_at_k` + tests
- [x] Security metrics (`src/rag/eval/security_metrics.py`) — `access_control_leakage`, `injection_defense_success_rate`, `pii_redaction_recall` + tests
- [ ] Generation metrics (RAGAS/DeepEval — faithfulness, answer relevance)
- [ ] Frozen gold set (50-100 hand-verified queries + relevant chunks + reference answers)
- [ ] Harness orchestration (run a pipeline config against the gold set, produce the benchmark table)

## 4. Retrieval ladder (each rung benchmarked against the frozen gold set)
- [ ] Naive dense top-k (baseline)
- [ ] + Hybrid (BM25 + RRF fusion)
- [ ] + Cross-encoder reranking
- [ ] + Contextual compression

## 5. Security layer + evals
- [ ] RBAC pre-filter (metadata filter *inside* the ANN query, never post-filter)
- [ ] Prompt-injection defense (hand-rolled)
- [ ] PII redaction on output (hand-rolled)

## 6. Bedrock integration
- [ ] Generation-model comparison axis
- [ ] Guardrails (PII + injection) benchmarked vs. hand-rolled
- [ ] Knowledge Bases — one managed-RAG baseline row

## 7. FastAPI serving
- [ ] API with role-based filtering injected from authenticated identity

## 8. Writeup
- [ ] README with benchmark tables (retrieval × generation model × managed-vs-custom security)
- [ ] "Techniques deliberately not used, and why" section
