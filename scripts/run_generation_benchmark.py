import json
import random
import sys

from rag.eval.harness import run_generation_eval
from rag.generation.bedrock_client import MODEL_IDS, generate_answer
from rag.observability.tracing import get_langfuse
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

    try:
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
    finally:
        get_langfuse().flush()


if __name__ == "__main__":
    main()
