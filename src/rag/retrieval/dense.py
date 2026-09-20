from sentence_transformers import SentenceTransformer
from rag.db.client import get_connection
from rag.security.access_filter import allowed_tiers

_model = None

def get_model():
    global _model
    if _model is None: 
        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _model

def naive_dense_retrieve(query: str, user_clearance: str, k: int = 10) -> list[str]:
    model = get_model()
    query_embedding = model.encode(query)
    tiers = allowed_tiers(user_clearance)

    conn = get_connection()
    cur = conn. execute(
        "SELECT chunk_id FROM chunks WHERE clearance_tier = ANY (%s) ORDER BY embedding <=> %s LIMIT %s",
        (tiers,query_embedding, k),
    )
    results = [row[0] for row in cur.fetchall()]
    conn.close()

    return results