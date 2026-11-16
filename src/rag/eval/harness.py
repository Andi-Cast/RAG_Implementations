import json
import re
from pathlib import Path
from typing import Callable

from rag.eval.generation_metrics import answer_correctness, answer_relevance, faithfulness
from rag.eval.retrieval_metrics import ndcg_at_k, reciprocal_rank, recall_at_k
from rag.retrieval.rerank import get_chunk_texts
from rag.security.injection_defense import sanitize_retrieved_context

# Same pattern as scripts/backfill_person_id.py -- a chunk_id's patient_id
# segment is encounter-scoped (Synthea appends an encounter-index number
# directly onto the end of the per-person UUID), so deriving the real
# person_id means matching the standard 8-4-4-4-12 hex UUID shape and
# dropping whatever digits were appended after it.
PERSON_ID_PATTERN = re.compile(
    r"^(.+?_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
)


def load_gold_set(path: str) -> list[dict]:
    """Load the gold set query entries from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return data["queries"]


def _derive_person_id(chunk_id: str) -> str:
    """Extract the real, stable person_id out of a chunk_id's
    patient_id::section prefix -- used to scope an eval to one person's
    full chart, a faithful simulation of a clinician already having that
    person's chart open before asking (not cheating with ground truth)."""
    patient_id = chunk_id.split("::")[0]
    match = PERSON_ID_PATTERN.match(patient_id)
    if match is None:
        raise ValueError(f"Could not derive person_id from chunk_id: {chunk_id!r}")
    return match.group(1)


def run_retrieval_eval(
    gold_set: list[dict],
    retrieve_fn: Callable[[str], list[str]],
    user_clearance: str,
    scope_to_person: bool = False,
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
        if scope_to_person:
            person_id = _derive_person_id(entry["relevant_chunk_ids"][0])
        else:
            person_id = None
        retrieved = retrieve_fn(entry["query"], user_clearance=user_clearance, person_id=person_id, k=k)

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
    scope_to_person: bool = False,
    k: int = 10,
) -> dict[str, float]:
    """For each gold-set query: retrieve context chunks via retrieve_fn
    (scoped to the query's own person when scope_to_person is True,
    same derivation as run_retrieval_eval -- a faithful simulation of a
    clinician already having that person's chart open before asking),
    sanitize each chunk's text via sanitize_retrieved_context (strips
    prompt-injection patterns before anything reaches the model), join
    the sanitized text into a single context string, generate an answer
    via generate_fn(query, context, model_id), score the answer with
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
        if scope_to_person:
            person_id = _derive_person_id(entry["relevant_chunk_ids"][0])
        else:
            person_id = None
        retrieved_ids = retrieve_fn(query, user_clearance=user_clearance, person_id=person_id, k=k)
        texts = get_chunk_texts(retrieved_ids)
        chunk_texts = [texts[chunk_id] for chunk_id in retrieved_ids]
        sanitized_texts = sanitize_retrieved_context(chunk_texts)
        context = "\n\n".join(sanitized_texts)

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
