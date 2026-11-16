CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    section TEXT NOT NULL,
    patient_id TEXT NOT NULL,
    -- patient_id is actually encounter-scoped (chunker.py derives it from
    -- the per-encounter export filename, Path(file_path).stem), not a
    -- stable per-person identifier -- Synthea exports one file per visit,
    -- so the same real person has a different patient_id per encounter.
    -- person_id is the real, stable identifier: patient_id's name prefix
    -- plus the first 36 chars of its UUID (standard 8-4-4-4-12 hex shape),
    -- discarding the encounter-index digits Synthea appends directly onto
    -- the end of the UUID with no separator. Backfilled via
    -- scripts/backfill_person_id.py, not populated at ingest time.
    person_id TEXT,
    clearance_tier TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL
);
