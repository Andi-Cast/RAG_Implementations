"""One-off backfill: derive the real, stable person_id from each chunk's
existing (encounter-scoped) patient_id, and write it into the new
person_id column.

patient_id is Path(file_path).stem from chunker.py -- Synthea exports one
file per encounter, and appends an encounter-index number directly onto
the end of its per-person UUID with no separator (e.g. the UUID
"...a81024f8526a" becomes "...a81024f8526a1" for one encounter and
"...a81024f8526a10" for another of the same real person). This strips
that trailing encounter index back off by matching the standard
8-4-4-4-12 hex UUID shape and keeping only up to its end.

Run with:
    uv run python scripts/backfill_person_id.py
"""

import re

from rag.db.client import get_connection

PERSON_ID_PATTERN = re.compile(
    r"^(.+?_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
)


def derive_person_id(patient_id: str) -> str | None:
    """Strip the encounter-index digits Synthea appends after the UUID,
    returning the stable per-person identifier. Returns None if
    patient_id doesn't match the expected name_UUID shape at all (should
    not happen against this corpus, but fail loud rather than silently
    writing a wrong value if the format is ever different)."""
    match = PERSON_ID_PATTERN.match(patient_id)
    return match.group(1) if match else None


def main() -> None:
    conn = get_connection()
    cur = conn.execute("SELECT DISTINCT patient_id FROM chunks")
    patient_ids = [row[0] for row in cur.fetchall()]

    unmatched = []
    updated = 0
    for patient_id in patient_ids:
        person_id = derive_person_id(patient_id)
        if person_id is None:
            unmatched.append(patient_id)
            continue
        conn.execute(
            "UPDATE chunks SET person_id = %s WHERE patient_id = %s",
            (person_id, patient_id),
        )
        updated += 1

    conn.commit()

    cur = conn.execute("SELECT COUNT(DISTINCT patient_id), COUNT(DISTINCT person_id) FROM chunks")
    n_patient_ids, n_person_ids = cur.fetchone()
    conn.close()

    print(f"Backfilled {updated} distinct patient_ids, {len(unmatched)} unmatched")
    if unmatched:
        print("Unmatched examples:", unmatched[:5])
    print(f"Distinct patient_id (encounters): {n_patient_ids}")
    print(f"Distinct person_id (real people): {n_person_ids}")


if __name__ == "__main__":
    main()
