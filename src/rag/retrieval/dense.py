from sentence_transformers import SentenceTransformer
from langfuse import observe
from rag.db.client import get_connection
from rag.observability.tracing import get_langfuse, mask_chunk_ids, mask_text
from rag.security.access_filter import allowed_tiers

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _model

@observe(name="retrieve-dense", as_type="retriever", capture_input=False, capture_output=False)
def naive_dense_retrieve(
    query: str, user_clearance: str, person_id: str | None, k: int = 10
) -> list[str]:
    model = get_model()
    query_embedding = model.encode(query)
    tiers = allowed_tiers(user_clearance)

    sql = "SELECT chunk_id FROM chunks WHERE clearance_tier = ANY(%s)"
    params = [tiers]
    if person_id is not None:
        sql += " AND person_id = %s"
        params.append(person_id)
    sql += " ORDER BY embedding <=> %s LIMIT %s"
    params += [query_embedding, k]

    conn = get_connection()
    cur = conn.execute(sql, params)
    results = [row[0] for row in cur.fetchall()]
    conn.close()

    get_langfuse().update_current_span(
        input={
            "query": mask_text(query),
            "user_clearance": user_clearance,
            "person_scoped": person_id is not None,
            "k": k,
        },
        output=mask_chunk_ids(results),
    )

    return results