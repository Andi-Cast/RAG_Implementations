"""Re-run the generation-model comparison against a random 50-query sample
of the 250-query gold set with person-scoped retrieval
(naive_dense_retrieve), replacing the original 14-query, unscoped pilot
documented in README.md. A fixed seed keeps the sample (and so the result)
reproducible across runs. Sampled rather than run against the full 250
because the AWS account's Bedrock on-demand rate limit throttles a full
1,000-call run (250 queries x 1 generate + 3 judge calls) to the point of
taking hours; 50 queries (200 calls) is the practical tradeoff between
statistical meaningfulness and finishing in a reasonable time.

    uv run python scripts/run_generation_benchmark.py haiku
    uv run python scripts/run_generation_benchmark.py gpt_oss
"""

import json
import random
import sys

from rag.eval.harness import run_generation_eval
from rag.generation.bedrock_client import MODEL_IDS, generate_answer
from rag.retrieval.dense import naive_dense_retrieve

SAMPLE_SIZE = 50
SAMPLE_SEED = 42


def main() -> None:
    model_key = sys.argv[1]
    model_id = MODEL_IDS[model_key]
    judge_model_id = MODEL_IDS["haiku"]

    with open("data/gold/gold_set.json") as f:
        gold_set = json.load(f)["queries"]

    gold_set = random.Random(SAMPLE_SEED).sample(gold_set, SAMPLE_SIZE)

    scores = run_generation_eval(
        gold_set=gold_set,
        retrieve_fn=naive_dense_retrieve,
        generate_fn=generate_answer,
        model_id=model_id,
        judge_model_id=judge_model_id,
        user_clearance="restricted",
        scope_to_person=True,
        k=10,
    )
    print(model_key, scores)


if __name__ == "__main__":
    main()
