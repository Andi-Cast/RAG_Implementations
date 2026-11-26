from langfuse import observe
from rag.db.client import get_connection
from rag.observability.tracing import get_langfuse, mask_chunk_ids, mask_text
from rag.security.access_filter import allowed_tiers


@observe(name="retrieve-sparse", as_type="retriever", capture_input=False, capture_output=False)
def sparse_retrieve(
    query: str, user_clearance: str, person_id: str | None, k: int = 10
) -> list[str]:
    tiers = allowed_tiers(user_clearance)
    conn = get_connection()

    sql = """
        SELECT chunk_id
        FROM chunks
        WHERE clearance_tier = ANY (%s)
        """
    params = [tiers]

    if person_id is not None:
        sql += " AND person_id = %s"
        params.append(person_id)

    sql += """
        AND to_tsvector('english', text) @@ to_tsquery(
        'english', regexp_replace(plainto_tsquery('english', %s)::text, ' & ', ' | ', 'g')
        )
        ORDER BY ts_rank(
            to_tsvector('english', text),
            to_tsquery('english', regexp_replace(plainto_tsquery('english', %s)::text, ' & ', ' | ', 'g'))
        ) DESC
        LIMIT %s;
        """

    params += [query, query, k]

    with conn.cursor() as cur:
        cur.execute(sql, params)
        results = cur.fetchall()
    chunk_ids = [row[0] for row in results]

    get_langfuse().update_current_span(
        input={
            "query": mask_text(query),
            "user_clearance": user_clearance,
            "person_scoped": person_id is not None,
            "k": k,
        },
        output=mask_chunk_ids(chunk_ids),
    )

    return chunk_ids