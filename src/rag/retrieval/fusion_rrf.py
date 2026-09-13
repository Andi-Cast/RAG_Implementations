def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
    top_k: int = 10,
) -> list[str]:
    """Merge multiple ranked lists of chunk_ids into one fused ranking.

    `rankings` is a list of ranked chunk_id lists (e.g. one from dense
    retrieval, one from sparse retrieval) -- each already ordered best-first.
    `k` is the RRF smoothing constant from the formula
    score(chunk) = sum over each ranking of 1 / (k + rank_in_that_ranking).
    `top_k` is how many chunk_ids to return from the fused ranking.

    Note: rank is 1-indexed (the first item in a ranking has rank 1, not 0).
    A chunk that doesn't appear in a given ranking contributes nothing from
    that ranking to its score.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank)

    ranked_chunk_ids = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return ranked_chunk_ids[:top_k]
