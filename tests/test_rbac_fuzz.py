import random

from rag.db.client import get_connection
from rag.models import CLEARANCE_TIERS
from rag.retrieval.dense import naive_dense_retrieve
from rag.security.access_filter import allowed_tiers

FUZZ_ITERATIONS = 30
FUZZ_SEED = 42


def _assert_no_leak(chunk_ids, user_clearance, person_id, conn):
    """Shared invariant check: every chunk naive_dense_retrieve returned
    must be within the caller's allowed tiers, and -- if a specific
    person_id was requested -- must belong to that person. Factored out
    so the single-case test and the randomized fuzz loop below both run
    the exact same check, rather than the loop re-deriving its own
    (possibly subtly different) version of it."""
    for chunk_id in chunk_ids:
        row = conn.execute(
            """
                SELECT clearance_tier, person_id
                FROM chunks
                WHERE chunk_id = %s
            """,
            (chunk_id,)
        ).fetchone()
        chunk_clearance_tier, chunk_person_id = row
        assert chunk_clearance_tier in allowed_tiers(user_clearance), f"Chunk {chunk_id} has clearance_tier {chunk_clearance_tier}, which is not allowed for user clearance {user_clearance}"
        if person_id is not None:
            assert chunk_person_id == person_id, f"Chunk {chunk_id} has person_id {chunk_person_id}, expected {person_id}"


def test_dense_retrieve_never_leaks_for_one_fixed_case():
    user_clearance = "internal"

    conn = get_connection()
    person_id = conn.execute(
        "SELECT person_id FROM chunks WHERE person_id IS NOT NULL LIMIT 1"
    ).fetchone()[0]

    chunk_ids = naive_dense_retrieve(
        query="What is the person's name?",
        person_id=person_id,
        user_clearance=user_clearance,
    )

    assert len(chunk_ids) > 0

    _assert_no_leak(chunk_ids, user_clearance, person_id, conn)
    conn.close()


def test_dense_retrieve_never_leaks_across_random_clearance_and_person_combos():
    """Same invariant as the fixed-case test above, but checked against
    FUZZ_ITERATIONS random (user_clearance, person_id) combinations
    instead of one hand-picked pair -- a regression that only shows up
    for some unusual combination (e.g. the lowest clearance tier paired
    with a specific person) is far more likely to surface this way than
    from a handful of hand-written examples. FUZZ_SEED keeps the random
    combinations reproducible across runs rather than flaking."""
    conn = get_connection()
    person_ids = [
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT person_id FROM chunks WHERE person_id IS NOT NULL LIMIT 50"
        ).fetchall()
    ]
    # None is included deliberately -- it's the explicit opt-in to
    # population-level search, and the leak check should hold for it too
    # (every returned chunk still has to respect clearance, even unscoped).
    person_id_choices = person_ids + [None]
    clearance_choices = list(CLEARANCE_TIERS.keys())

    rng = random.Random(FUZZ_SEED)
    for _ in range(FUZZ_ITERATIONS):
        user_clearance = rng.choice(clearance_choices)
        person_id = rng.choice(person_id_choices)

        chunk_ids = naive_dense_retrieve(
            query="What is the person's name?",
            person_id=person_id,
            user_clearance=user_clearance,
        )
        _assert_no_leak(chunk_ids, user_clearance, person_id, conn)

    conn.close()