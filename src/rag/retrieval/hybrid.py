from rag.retrieval.dense import naive_dense_retrieve
from rag.retrieval.fusion_rrf import reciprocal_rank_fusion
from rag.retrieval.sparse_bm25 import sparse_retrieve


def hybrid_retrieve(query: str, user_clearance: str, k: int = 10, candidate_k: int = 50) -> list[str]:
    """Rung 2 of the retrieval ladder: run both naive_dense_retrieve and
    sparse_retrieve (each pulling a wider candidate_k pool), then merge
    their rankings with reciprocal_rank_fusion and trim to the final k.
    Matches the retrieve_fn signature run_retrieval_eval expects.
    """

    dense_results = naive_dense_retrieve(query, user_clearance=user_clearance, k=candidate_k)
    sparse_results = sparse_retrieve(query, user_clearance=user_clearance, k=candidate_k)
    return reciprocal_rank_fusion([dense_results, sparse_results], k=60, top_k=k)
