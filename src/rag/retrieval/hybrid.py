from langfuse import observe
from rag.observability.tracing import get_langfuse, mask_chunk_ids, mask_text
from rag.retrieval.dense import naive_dense_retrieve
from rag.retrieval.fusion_rrf import reciprocal_rank_fusion
from rag.retrieval.sparse_bm25 import sparse_retrieve


@observe(name="retrieve-hybrid", as_type="retriever", capture_input=False, capture_output=False)
def hybrid_retrieve(query: str, user_clearance: str, person_id: str | None, k: int = 10, candidate_k: int = 50) -> list[str]:
    dense_results = naive_dense_retrieve(query, user_clearance=user_clearance, person_id=person_id, k=candidate_k)
    sparse_results = sparse_retrieve(query, user_clearance=user_clearance, person_id=person_id, k=candidate_k)
    results = reciprocal_rank_fusion([dense_results, sparse_results], k=60, top_k=k)

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
