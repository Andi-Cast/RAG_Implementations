from sentence_transformers import SentenceTransformer
from rag.db.client import get_connection
from rag.security.access_filter import allowed_tiers

_model = None

def get_model():
    global _model
    if _model is None: 
        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _model

def naive_dense_retrieve(
    query: str, user_clearance: str, person_id: str | None, k: int = 10
) -> list[str]:
    """person_id has no default -- callers must explicitly decide between
    scoping to one real person's full chart across all their encounters
    (the realistic clinical workflow) or passing None to deliberately
    search across all people (population-level/research queries). Same
    fail-loud-not-fail-open reasoning as user_clearance, just allowing
    None as a conscious opt-out rather than a silent one.

    person_id is distinct from the chunks table's own patient_id column,
    which is actually encounter-scoped (one real person has a different
    patient_id per visit) -- see schema.sql and
    scripts/backfill_person_id.py for why."""
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

    return results