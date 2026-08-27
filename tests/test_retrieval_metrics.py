import pytest

from rag.eval.retrieval_metrics import ndcg_at_k, reciprocal_rank, recall_at_k


def test_recall_at_k_partial_match():
    retrieved = ["c1", "c2", "c3", "c4"]
    relevant = {"c2", "c5"}
    assert recall_at_k(retrieved, relevant, k=3) == 0.5


def test_recall_at_k_no_relevant_docs():
    assert recall_at_k(["c1", "c2"], set(), k=2) == 0.0


def test_recall_at_k_k_larger_than_retrieved():
    retrieved = ["c1", "c2"]
    relevant = {"c1"}
    assert recall_at_k(retrieved, relevant, k=10) == 1.0


def test_reciprocal_rank_hit_at_second_position():
    retrieval = ["c1", "c2", "c3"]
    relevant = {"c2", "c5"}
    assert reciprocal_rank(retrieval, relevant) == 0.5


def test_reciprocal_rank_hit_at_first_position():
    retrieval = ["c1", "c2", "c3"]
    relevant = {"c1"}
    assert reciprocal_rank(retrieval, relevant) == 1.0


def test_reciprocal_rank_no_hit():
    retrieval = ["c1", "c2", "c3"]
    relevant = {"c9"}
    assert reciprocal_rank(retrieval, relevant) == 0.0


def test_ndcg_at_k_matches_worked_example():
    retrieved = ["c3", "c1", "c2"]
    relevance = {"c1": 2.0, "c2": 1.0}
    assert ndcg_at_k(retrieved, relevance, k=3) == pytest.approx(0.670, abs=1e-3)


def test_ndcg_at_k_respects_k_cutoff():
    retrieved = ["c3", "c1", "c2"]
    relevance = {"c1": 2.0, "c2": 1.0}
    assert ndcg_at_k(retrieved, relevance, k=1) == 0.0


def test_ndcg_at_k_perfect_ranking_scores_one():
    retrieved = ["c1", "c2"]
    relevance = {"c1": 2.0, "c2": 1.0}
    assert ndcg_at_k(retrieved, relevance, k=2) == pytest.approx(1.0)


def test_ndcg_at_k_no_relevance_info():
    assert ndcg_at_k(["c1", "c2"], {}, k=2) == 0.0
