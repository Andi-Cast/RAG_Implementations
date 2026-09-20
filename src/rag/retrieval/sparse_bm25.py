from rag.db.client import get_connection
from rag.security.access_filter import allowed_tiers


def sparse_retrieve(query: str, user_clearance: str, k: int = 10) -> list[str]:
    """Keyword-based retrieval using Postgres full-text search.

    Hint: `to_tsvector('english', text)` turns chunk text into a searchable
    document, `to_tsquery('english', ...)` turns the query into a search
    term, and `ts_rank(...)` scores how well each chunk matches. Order by
    that rank descending, limit to k, and return the chunk_ids.
    """
    tiers = allowed_tiers(user_clearance)
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT chunk_id
            FROM chunks
            WHERE clearance_tier = ANY (%s)
            AND to_tsvector('english', text) @@ to_tsquery(
                'english', regexp_replace(plainto_tsquery('english', %s)::text, ' & ', ' | ', 'g')
            )
            ORDER BY ts_rank(
                to_tsvector('english', text),
                to_tsquery('english', regexp_replace(plainto_tsquery('english', %s)::text, ' & ', ' | ', 'g'))
            ) DESC
            LIMIT %s;
            """,
            (tiers, query, query, k),
        )
        results = cur.fetchall()
    return [row[0] for row in results]