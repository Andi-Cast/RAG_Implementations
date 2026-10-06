import json
from pathlib import Path
from typing import Callable

from rag.eval.generation_metrics import answer_correctness, answer_relevance, faithfulness
from rag.eval.retrieval_metrics import ndcg_at_k, reciprocal_rank, recall_at_k
from rag.retrieval.rerank import get_chunk_texts


def load_gold_set(path: str) -> list[dict]:
    """Load the gold set query entries from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return data["queries"]


def run_retrieval_eval(
    gold_set: list[dict],
    retrieve_fn: Callable[[str], list[str]],
    user_clearance: str,
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
        retrieved = retrieve_fn(entry["query"], user_clearance=user_clearance, k=k)

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


def run_generation_eval(
    gold_set: list[dict],
    retrieve_fn: Callable[..., list[str]],
    generate_fn: Callable[[str, str, str], str],
    model_id: str,
    judge_model_id: str,
    user_clearance: str,
    k: int = 10,
) -> dict[str, float]:
    """For each gold-set query: retrieve context chunks via retrieve_fn,
    join their text into a single context string, generate an answer via
    generate_fn(query, context, model_id), score the answer with
    faithfulness(context, answer, judge_model_id),
    answer_relevance(query, answer, judge_model_id), and
    answer_correctness(entry["reference_answer"], answer, judge_model_id),
    and return the averaged scores across the whole gold set. Scores
    faithfulness against the context actually retrieved and fed to the
    model (not the gold set's ground-truth relevant_chunk_ids), since
    faithfulness measures whether the model stuck to what it was actually
    given -- which means this convolves retrieval quality with generation
    quality by design. answer_correctness is the one metric here that
    isn't reference-free -- it's what catches a retrieval failure that
    faithfulness/relevance alone would miss."""
    faithfulness_scores = []
    relevance_scores = []
    correctness_scores = []

    for entry in gold_set:
        query = entry["query"]
        retrieved_ids = retrieve_fn(query, user_clearance=user_clearance, k=k)
        texts = get_chunk_texts(retrieved_ids)
        context = "\n\n".join(texts[chunk_id] for chunk_id in retrieved_ids)

        answer = generate_fn(query, context, model_id)

        faithfulness_scores.append(faithfulness(context, answer, judge_model_id))
        relevance_scores.append(answer_relevance(query, answer, judge_model_id))
        correctness_scores.append(
            answer_correctness(entry["reference_answer"], answer, judge_model_id)
        )

    n = len(gold_set)
    return {
        "faithfulness": sum(faithfulness_scores) / n,
        "answer_relevance": sum(relevance_scores) / n,
        "answer_correctness": sum(correctness_scores) / n,
    }
