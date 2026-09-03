CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    text TEXT NOT NULL, 
    section TEXT NOT NULL, 
    patient_id TEXT NOT NULL, 
    clearance_tier TEXT NOT NULL, 
    embedding VECTOR(384) NOT NULL
);
