"""Build a small, committed CI fixture: a handful of real people's chunks
(with embeddings) plus the subset of the gold set that's actually about
them, so GitHub Actions can run a real retrieval-regression check without
restoring or re-embedding the full 92k-chunk corpus on every run.

    uv run python scripts/build_ci_fixture.py
"""
import gzip
import json
import subprocess
from collections import Counter

from rag.db.client import get_connection
from rag.eval.harness import _derive_person_id

CI_FIXTURE_PERSON_COUNT = 5
# The corpus averages ~393 chunks/person (91,969 chunks / 234 people).
# Capping candidates at ~1.5x that excludes the small number of outlier
# "mega-patients" with thousands of chunks -- ranking by gold-set query
# frequency alone (with no cap) picks exactly those outliers, since an
# unusually long medical history is also why they generated so many
# gold-set queries in the first place, which blows up the fixture dump's
# size for little real benefit (one typical-sized person's queries are
# just as good a regression signal as one mega-patient's).
CI_FIXTURE_MAX_CHUNKS_PER_PERSON = 600
CI_FIXTURE_TABLE = "chunks_ci_fixture"
DOCKER_COMPOSE_POSTGRES_SERVICE = "postgres"
POSTGRES_USER = "rag"
POSTGRES_DB = "rag"


def pick_fixture_person_ids(gold_set: list[dict], count: int, conn) -> list[str]:
    """Return the `count` person_ids with the MOST gold-set queries
    attached to them (derived the same way harness.py's
    _derive_person_id does, from each entry's relevant_chunk_ids[0]),
    among only those whose total chunk count is at or below
    CI_FIXTURE_MAX_CHUNKS_PER_PERSON -- ranking by frequency alone, with
    no size cap, tends to pick exactly the corpus's outlier mega-patients
    (see CI_FIXTURE_MAX_CHUNKS_PER_PERSON's comment for why), which
    maximizes gold-set coverage at the cost of a much larger fixture
    dump for little real benefit."""
    person_id_counts = Counter(
        _derive_person_id(entry["relevant_chunk_ids"][0])
        for entry in gold_set
        if entry.get("relevant_chunk_ids")
    )

    picked = []
    for person_id, _ in person_id_counts.most_common():
        if len(picked) >= count:
            break
        chunk_count = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE person_id = %s", (person_id,)
        ).fetchone()[0]
        if chunk_count <= CI_FIXTURE_MAX_CHUNKS_PER_PERSON:
            picked.append(person_id)

    return sorted(picked)


def export_fixture_dump(person_ids: list[str], output_path: str) -> None:
    """Export just person_ids' chunk rows (schema + embeddings included)
    into a gzip-compressed SQL dump at output_path, so CI can restore a
    small, committed fixture instead of the full corpus. pg_dump has no
    row-filtering flag, so the standard trick is: build a temp table
    holding only the rows we want, dump *that* table with --data-only,
    then rewrite the dump's table name back to the real one -- dumping
    runs inside the running `postgres` docker-compose service itself
    (via `docker compose exec`), since this machine has no local pg_dump/
    psql client, and that also guarantees the dump tool's version always
    matches the server's. Gzipped because each person's full chart
    (embeddings included) is several MB as plain text -- uncompressed,
    even 5 people would be a large file to permanently commit to git."""
    conn = get_connection()
    conn.execute(f"DROP TABLE IF EXISTS {CI_FIXTURE_TABLE}")
    conn.execute(
        f"CREATE TABLE {CI_FIXTURE_TABLE} AS "
        f"SELECT * FROM chunks WHERE person_id = ANY(%s)",
        (person_ids,),
    )
    conn.commit()

    try:
        result = subprocess.run(
            [
                "docker", "compose", "exec", "-T", DOCKER_COMPOSE_POSTGRES_SERVICE,
                "pg_dump", "-U", POSTGRES_USER, "-d", POSTGRES_DB,
                "--data-only", f"--table={CI_FIXTURE_TABLE}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    finally:
        conn.execute(f"DROP TABLE IF EXISTS {CI_FIXTURE_TABLE}")
        conn.commit()
        conn.close()

    dump_sql = result.stdout.replace(CI_FIXTURE_TABLE, "chunks")
    with gzip.open(output_path, "wt") as f:
        f.write(dump_sql)


def build_ci_gold_sample(gold_set: list[dict], person_ids: list[str]) -> list[dict]:
    """Return the subset of gold_set entries whose derived person_id (via
    _derive_person_id, same as pick_fixture_person_ids) is in person_ids --
    the matching slice of real, meaningful queries for the people actually
    included in the committed fixture dump.
    TODO: implement.
    """
    raise NotImplementedError


def main() -> None:
    with open("data/gold/gold_set.json") as f:
        gold_set = json.load(f)["queries"]

    conn = get_connection()
    person_ids = pick_fixture_person_ids(gold_set, CI_FIXTURE_PERSON_COUNT, conn)
    conn.close()
    print(f"Picked {len(person_ids)} fixture person_ids: {person_ids}")

    output_path = "tests/fixtures/ci_corpus.sql.gz"
    export_fixture_dump(person_ids, output_path)
    print(f"Wrote fixture dump to {output_path}")


if __name__ == "__main__":
    main()
