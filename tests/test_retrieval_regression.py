import json

from rag.eval.harness import run_retrieval_eval
from rag.retrieval.dense import naive_dense_retrieve

BASELINE_RECALL_AT_10 = 0.45
BASELINE_MRR = 0.20
BASELINE_NDCG_AT_10 = 0.26


def test_naive_dense_retrieve_meets_baseline_on_ci_fixture():
    with open("tests/fixtures/ci_gold_sample.json") as f:
        gold_set = json.load(f)["queries"]

    scores = run_retrieval_eval(
        gold_set=gold_set,
        retrieve_fn=naive_dense_retrieve,
        user_clearance="restricted",
        scope_to_person=True,
        k=10,
    )

    assert scores["recall_at_10"] >= BASELINE_RECALL_AT_10
    assert scores["mrr"] >= BASELINE_MRR
    assert scores["ndcg_at_10"] >= BASELINE_NDCG_AT_10
