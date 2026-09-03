from sentence_transformers import SentenceTransformer
from rag.db.client import get_connection

_model = None

def _get_model():
    global _model
    if _model is None: 
        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _model

def naive_dense_retrieve(query: str, k: int = 10) -> list[str]:
    model = _get_model()
    query_embedding = model.encode(query)

    conn = get_connection()
    cur = conn. execute(
        "SELECT chunk_id FROM chunks ORDER BY embedding <=> %s LIMIT %s",
        (query_embedding, k),
    )
    results = [row[0] for row in cur.fetchall()]
    conn.close()

    return results