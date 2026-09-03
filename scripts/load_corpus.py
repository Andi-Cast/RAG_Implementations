import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import glob

from sentence_transformers import SentenceTransformer

from rag.ingest.chunker import chunk_encounter_file
from rag.ingest.clearance_tagger import tag_clearance_tier
from rag.models import Chunk

def load_all_chunks(directory: str) -> list[Chunk]:
    all_chunks = []
    for file_path in glob.glob(f"{directory}/*.txt"):
        chunks = chunk_encounter_file(file_path)
        for chunk in chunks:
            tag_clearance_tier(chunk)
        all_chunks.extend(chunks)
    return all_chunks

def embed_chunks(chunks: list[Chunk], model: SentenceTransformer) -> list:
    texts = [chunk.text for chunk in chunks]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    return embeddings

def insert_chunks(conn, chunks: list[Chunk], embeddings) -> None:
    params = [
        (chunk.chunk_id, 
         chunk.text,
         chunk.section,
         chunk.patient_id,
         chunk.clearance_tier, 
         embedding)
        for chunk, embedding in zip(chunks, embeddings)
    ]

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks (chunk_id, text, section, patient_id, clearance_tier, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO NOTHING
            """,
            params,
        )
    conn.commit()

from rag.db.client import get_connection

def main():
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    chunks = load_all_chunks("synthea/output/text_encounters")
    embeddings = embed_chunks(chunks, model)

    conn = get_connection()
    insert_chunks(conn, chunks, embeddings)
    conn.close()

    print(f"Loaded {len(chunks)} chunks into the database.")

if __name__ == "__main__":
    main()
