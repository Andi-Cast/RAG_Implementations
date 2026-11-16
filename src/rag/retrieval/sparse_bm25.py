from rag.db.client import get_connection
from rag.security.access_filter import allowed_tiers


def sparse_retrieve(
    query: str, user_clearance: str, person_id: str | None, k: int = 10
) -> list[str]:
    """Keyword-based retrieval using Postgres full-text search.

    person_id has no default -- pass None to deliberately search across
    all people, or a specific person_id to scope to one person's full
    chart across all their encounters. Distinct from the chunks table's
    own patient_id column, which is actually encounter-scoped -- see
    schema.sql and scripts/backfill_person_id.py.

    Uses `to_tsvector('english', text)` to turn chunk text into a
    searchable document, `to_tsquery('english', ...)` to turn the query
    into a search term, and `ts_rank(...)` to score how well each chunk
    matches, ordering by that rank descending.
    """
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
    return [row[0] for row in results]