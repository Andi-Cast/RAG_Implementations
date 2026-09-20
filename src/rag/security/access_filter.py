from rag.models import CLEARANCE_TIERS


def allowed_tiers(user_clearance: str) -> list[str]:
    """Return every clearance tier name at or below user_clearance's rank
    in CLEARANCE_TIERS -- e.g. an "internal" user gets
    ["public", "internal"], not "restricted" or "confidential".

    This is what gets passed to `clearance_tier = ANY(%s)` in a retrieval
    query's WHERE clause -- the actual pre-filter enforcement happens at
    the SQL level, not by filtering results after the fact.
    """
    user_rank = CLEARANCE_TIERS[user_clearance]
    return [tier for tier, rank in CLEARANCE_TIERS.items() if rank <= user_rank]
