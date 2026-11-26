from sentence_transformers import CrossEncoder
from langfuse import observe

from rag.db.client import get_connection
from rag.observability.tracing import get_langfuse, mask_chunk_ids, mask_text
from rag.retrieval.hybrid import hybrid_retrieve

_reranker = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

@observe(name="get-chunk-texts", as_type="retriever", capture_input=False, capture_output=False)
def get_chunk_texts(chunk_ids: list[str]) -> dict[str, str]:
    conn = get_connection()
    cur = conn.execute(
        "SELECT chunk_id, text FROM chunks WHERE chunk_id = ANY(%s)",
        (chunk_ids,),
    )
    result = dict(cur.fetchall())
    conn.close()

    get_langfuse().update_current_span(
        input={"chunk_ids": mask_chunk_ids(chunk_ids)},
        output={"chunk_texts": mask_chunk_ids(list(result.keys()))},
    )

    return result


@observe(name="rerank-candidates", as_type="retriever", capture_input=False, capture_output=False)
def cross_encoder_rerank(query: str, user_clearance: str, person_id: str | None, k: int = 10, candidate_k: int = 50) -> list[str]:
    candidate_ids = hybrid_retrieve(query, user_clearance=user_clearance, person_id=person_id, k=candidate_k)
    texts = get_chunk_texts(candidate_ids)

    model = _get_reranker()
    pairs = [(query, texts[chunk_id]) for chunk_id in candidate_ids]
    scores = model.predict(pairs)

    ranked = sorted(zip(candidate_ids, scores), key=lambda pair: pair[1], reverse=True)
    results = [chunk_id for chunk_id, _score in ranked[:k]]

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
