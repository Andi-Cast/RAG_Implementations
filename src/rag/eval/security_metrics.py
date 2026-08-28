def access_control_leakage(
    returned_clearances: list[str], 
    user_clearance: str, 
    tier_order: dict[str, int],
) -> float:
    if not returned_clearances:
        return 0.0
    
    num_of_leakages = 0

    for clearance in returned_clearances:
        if tier_order[clearance] > tier_order[user_clearance]:
            num_of_leakages += 1

    return num_of_leakages / len(returned_clearances)

def injection_defense_success_rate(blocked_flags: list[bool]) -> float: 
    if not blocked_flags:
        return 0.0
    
    num_of_passes = sum(blocked_flags)

    return num_of_passes / len(blocked_flags)

def pii_redaction_recall(true_spans: set[str], redacted_spans: set[str]) -> float:
    if not true_spans:
        return 1.0

    overlap = true_spans & redacted_spans

    return len(overlap) / len(true_spans)


