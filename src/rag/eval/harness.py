import json
from pathlib import Path
from typing import Callable

from rag.eval.retrieval_metrics import ndcg_at_k, reciprocal_rank, recall_at_k


def load_gold_set(path: str) -> list[dict]:
    """Load the gold set query entries from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return data["queries"]


def run_retrieval_eval(
    gold_set: list[dict],
    retrieve_fn: Callable[[str], list[str]],
    k: int = 10,
) -> dict[str, float]:
    """Run retrieve_fn against every query in the gold set, score each
    result against its relevant_chunk_ids, and return the averaged
    retrieval metrics across the whole gold set."""

    recall_scores = []
    rr_scores = []
    ndcg_scores = []

    for entry in gold_set:
        relevant = set(entry["relevant_chunk_ids"])
        retrieved = retrieve_fn(entry["query"])

        relevance = {chunk_id: 1.0 for chunk_id in relevant}

        recall_scores.append(recall_at_k(retrieved, relevant, k))
        rr_scores.append(reciprocal_rank(retrieved, relevant))
        ndcg_scores.append(ndcg_at_k(retrieved, relevance, k))

    n = len(gold_set)
    return {
        f"recall_at_{k}": sum(recall_scores) / n,
        "mrr": sum(rr_scores) / n,
        f"ndcg_at_{k}": sum(ndcg_scores) / n,
    }
