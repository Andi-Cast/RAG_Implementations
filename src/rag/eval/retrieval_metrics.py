from math import log2

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0

    top_k_list = retrieved[:k]

    count = sum(1 for item in relevant if item in top_k_list)

    return count / len(relevant)

def reciprocal_rank(retrieval: list[str], relevant: list[str]) -> float:
    for i, item in enumerate(retrieval, start = 1):
        if item in relevant:
            return 1 / i
           
    return 0.0

def ndcg_at_k(retrieved: list[str], relevance: dict[str, float], k: int) -> float: 
    if not retrieved:
        return 0.0

    dcg = 0
    for i, value in enumerate(retrieved[:k]):
        if value in relevance:
            dcg += relevance[value] / log2(i + 2)

    list_by_value = sorted(relevance.items(), key = lambda x:x[1], reverse=True)
    top_k = list_by_value[:k]

    idcg = 0
    for i, x in enumerate(top_k):
        idcg += x[1] / log2(i + 2)

    if idcg == 0:
        return 0.0
    return dcg / idcg
