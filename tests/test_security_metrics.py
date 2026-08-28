import pytest
from rag.eval.security_metrics import access_control_leakage, injection_defense_success_rate, pii_redaction_recall

tier_rank = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}


def test_access_control_leakage_leak_case():
    returned_clearances = ['restricted']
    user_clearance = 'public'
    tier_order = tier_rank
    assert access_control_leakage(returned_clearances, user_clearance, tier_order) == 1.0

def test_access_control_leakage_no_leak_case():
    returned_clearances = ['restricted']
    user_clearance = 'restricted'
    tier_order = tier_rank
    assert access_control_leakage(returned_clearances, user_clearance, tier_order) == 0.0

def test_access_control_leakage_mixed_leak_case():
    returned_clearances = ['restricted', 'public']
    user_clearance = 'public'
    tier_order = tier_rank
    assert access_control_leakage(returned_clearances, user_clearance, tier_order) == 0.5

def test_access_control_leakage_empty_list():
    returned_clearances = []
    user_clearance = 'public'
    tier_order = tier_rank
    assert access_control_leakage(returned_clearances, user_clearance, tier_order) == 0.0

def test_injection_defense_success_rate_pass():
    blocked_flags = [True, True, True]
    assert injection_defense_success_rate(blocked_flags) == 1.0

def test_injection_defense_success_rate_mix():
    blocked_flags = [True, False, False, True]
    assert injection_defense_success_rate(blocked_flags) ==  0.5

def test_injection_defense_success_rate_fail():
    blocked_flags = [False, False, False]
    assert injection_defense_success_rate(blocked_flags) == 0.0

def test_injection_defense_success_rate_empty_list():
    blocked_flags = []
    assert injection_defense_success_rate(blocked_flags) == 0.0

def test_pii_redaction_recall_pass():
    true_spans = {"test1", "test2"}
    redacted_spans = {"test1", "test2"}
    assert pii_redaction_recall(true_spans, redacted_spans) == 1.0

def test_pii_redaction_recall_fail():
    true_spans = {"test1", "test2"}
    redacted_spans = {"test3"}
    assert pii_redaction_recall(true_spans, redacted_spans) == 0.0

def test_pii_redaction_recall_mixed():
    true_spans = {"span_1", "span_2", "span_3"}
    redacted_spans = {"span_1"}
    assert pii_redaction_recall(true_spans, redacted_spans) == pytest.approx(1 / 3)

def test_pii_redaction_recall_empty_true_spans():
    true_spans = set()
    redacted_spans = set()
    assert pii_redaction_recall(true_spans, redacted_spans) == 1.0
