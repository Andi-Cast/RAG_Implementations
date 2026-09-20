from sentence_transformers import CrossEncoder

from rag.db.client import get_connection
from rag.retrieval.hybrid import hybrid_retrieve

_reranker = None


def _get_reranker() -> CrossEncoder:
    """Lazily load and cache the cross-encoder model (same lazy-singleton
    pattern as _get_model() in dense.py)."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def get_chunk_texts(chunk_ids: list[str]) -> dict[str, str]:
    """Fetch the text for a list of chunk_ids, returned as
    {chunk_id: text}."""
    conn = get_connection()
    cur = conn.execute(
        "SELECT chunk_id, text FROM chunks WHERE chunk_id = ANY(%s)",
        (chunk_ids,),
    )
    result = dict(cur.fetchall())
    conn.close()
    return result


def cross_encoder_rerank(query: str, user_clearance: str, k: int = 10, candidate_k: int = 50) -> list[str]:
    """Rung 3 of the retrieval ladder: pull a candidate_k-sized pool from
    hybrid_retrieve, score each (query, chunk_text) pair with the
    cross-encoder, and return the top-k chunk_ids by that score.
    Matches the retrieve_fn signature run_retrieval_eval expects.
    """
    candidate_ids = hybrid_retrieve(query, user_clearance=user_clearance, k=candidate_k)
    texts = get_chunk_texts(candidate_ids)

    model = _get_reranker()
    pairs = [(query, texts[chunk_id]) for chunk_id in candidate_ids]
    scores = model.predict(pairs)

    ranked = sorted(zip(candidate_ids, scores), key=lambda pair: pair[1], reverse=True)
    return [chunk_id for chunk_id, _score in ranked[:k]]
