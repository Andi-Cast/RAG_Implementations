import json
import re
from pathlib import Path
from typing import Callable

from langfuse import propagate_attributes

from rag.eval.generation_metrics import answer_correctness, answer_relevance, faithfulness
from rag.eval.retrieval_metrics import ndcg_at_k, reciprocal_rank, recall_at_k
from rag.observability.tracing import get_langfuse, mask_text
from rag.retrieval.compression import compress_chunks
from rag.retrieval.rerank import get_chunk_texts
from rag.security.injection_defense import sanitize_retrieved_context
from rag.security.pii_redaction import redact_pii

PERSON_ID_PATTERN = re.compile(
    r"^(.+?_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
)


def load_gold_set(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["queries"]


def _derive_person_id(chunk_id: str) -> str:
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
    recall_scores = []
    rr_scores = []
    ndcg_scores = []
    langfuse = get_langfuse()

    for entry in gold_set:
        with langfuse.start_as_current_observation(
            as_type="span",
            name="evaluate-retrieval-query",
            input={"query_id": entry.get("query_id"), "query": mask_text(entry["query"])},
        ) as root, propagate_attributes(
            environment="development",
            tags=["retrieval-eval"],
            metadata={"user_clearance": user_clearance, "scope_to_person": scope_to_person},
        ):
            relevant = set(entry["relevant_chunk_ids"])
            if scope_to_person:
                person_id = _derive_person_id(entry["relevant_chunk_ids"][0])
            else:
                person_id = None
            retrieved = retrieve_fn(entry["query"], user_clearance=user_clearance, person_id=person_id, k=k)

            relevance = {chunk_id: 1.0 for chunk_id in relevant}

            recall = recall_at_k(retrieved, relevant, k)
            rr = reciprocal_rank(retrieved, relevant)
            ndcg = ndcg_at_k(retrieved, relevance, k)
            recall_scores.append(recall)
            rr_scores.append(rr)
            ndcg_scores.append(ndcg)

            root.update(output={f"recall_at_{k}": recall, "mrr": rr, f"ndcg_at_{k}": ndcg})

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
    compress: bool = False,
    k: int = 10,
) -> dict[str, float]:
    faithfulness_scores = []
    relevance_scores = []
    correctness_scores = []
    langfuse = get_langfuse()

    for entry in gold_set:
        query = entry["query"]
        with langfuse.start_as_current_observation(
            as_type="span",
            name="evaluate-generation-query",
            input={"query_id": entry.get("query_id"), "query": mask_text(query)},
        ) as root, propagate_attributes(
            environment="development",
            tags=["generation-eval"],
            metadata={
                "user_clearance": user_clearance,
                "scope_to_person": scope_to_person,
                "model_id": model_id,
                "judge_model_id": judge_model_id,
            },
        ):
            if scope_to_person:
                person_id = _derive_person_id(entry["relevant_chunk_ids"][0])
            else:
                person_id = None
            retrieved_ids = retrieve_fn(query, user_clearance=user_clearance, person_id=person_id, k=k)
            if compress:
                texts = compress_chunks(query, retrieved_ids)
            else:
                texts = get_chunk_texts(retrieved_ids)
            chunk_texts = [texts[chunk_id] for chunk_id in retrieved_ids]
            sanitized_texts = sanitize_retrieved_context(chunk_texts)
            context = "\n\n".join(sanitized_texts)

            answer = generate_fn(query, context, model_id)

            faithfulness_score = faithfulness(context, answer, judge_model_id)
            relevance_score = answer_relevance(query, answer, judge_model_id)
            correctness_score = answer_correctness(entry["reference_answer"], answer, judge_model_id)
            faithfulness_scores.append(faithfulness_score)
            relevance_scores.append(relevance_score)
            correctness_scores.append(correctness_score)

            redacted_answer = redact_pii(answer)
            root.update(output={
                "answer": mask_text(redacted_answer, 200),
                "faithfulness": faithfulness_score,
                "answer_relevance": relevance_score,
                "answer_correctness": correctness_score,
            })

    n = len(gold_set)
    return {
        "faithfulness": sum(faithfulness_scores) / n,
        "answer_relevance": sum(relevance_scores) / n,
        "answer_correctness": sum(correctness_scores) / n,
    }
