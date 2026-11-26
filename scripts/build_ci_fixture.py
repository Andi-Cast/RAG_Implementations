import gzip
import json
import subprocess
from collections import Counter

from rag.db.client import get_connection
from rag.eval.harness import _derive_person_id

CI_FIXTURE_PERSON_COUNT = 5
CI_FIXTURE_MAX_CHUNKS_PER_PERSON = 600
CI_FIXTURE_TABLE = "chunks_ci_fixture"
DOCKER_COMPOSE_POSTGRES_SERVICE = "postgres"
POSTGRES_USER = "rag"
POSTGRES_DB = "rag"


def pick_fixture_person_ids(gold_set: list[dict], count: int, conn) -> list[str]:
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
    dump_sql = "\n".join(
        line for line in dump_sql.splitlines()
        if not line.startswith("\\restrict") and not line.startswith("\\unrestrict")
    ) + "\n"
    with gzip.GzipFile(output_path, "wb", mtime=0) as gz:
        gz.write(dump_sql.encode())


def build_ci_gold_sample(gold_set: list[dict], person_ids: list[str]) -> list[dict]:
    person_id_set = set(person_ids)
    return [
        entry
        for entry in gold_set
        if entry.get("relevant_chunk_ids")
        and _derive_person_id(entry["relevant_chunk_ids"][0]) in person_id_set
    ]


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

    gold_sample = build_ci_gold_sample(gold_set, person_ids)
    gold_sample_path = "tests/fixtures/ci_gold_sample.json"
    with open(gold_sample_path, "w") as f:
        json.dump({"queries": gold_sample}, f, indent=2)
    print(f"Wrote {len(gold_sample)} gold-set queries to {gold_sample_path}")


if __name__ == "__main__":
    main()
