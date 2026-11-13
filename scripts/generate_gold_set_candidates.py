"""Expand the gold set from 14 hand-drafted queries toward 150, using a
deterministic template-based generator -- no LLM calls, no hallucination
risk, every answer is read directly off the chunk's own structured text.
Same "hand-rolled, inspectable" approach as the rest of this project.

Run with:
    uv run python scripts/generate_gold_set_candidates.py
"""

import json
import re

from rag.db.client import get_connection

GOLD_SET_PATH = "data/gold/gold_set.json"
TARGET_TOTAL = 250
BATCH_SIZE = 20

# Sections whose first line follows "date : description (type)" --
# REPORTS and CARE PLANS also match on their top line; nested detail
# lines (lab components, Reason:/Activity:) are ignored.
SIMPLE_SECTIONS = {
    "CONDITIONS": "What condition was diagnosed for this patient on {date}?",
    "PROCEDURES": "What procedure was performed on this patient on {date}?",
    "MEDICATIONS": "What medication was prescribed to this patient on {date}?",
    "IMMUNIZATIONS": "What immunization did this patient receive on {date}?",
    "IMAGING STUDIES": "What imaging study was performed on this patient on {date}?",
    "CARE PLANS": "What care plan was created for this patient on {date}?",
    "REPORTS": "What report was generated for this patient on {date}?",
}

DATE_LINE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})\s*:\s*(.+)")
TRAILING_TYPE_TAG = re.compile(r"\s*\([^)]*\)\s*$")
# Anchored per-line (callers apply this to one line at a time via .match,
# not .search on the whole multi-line chunk) -- an earlier version used
# $ against the whole chunk text and let group 3 greedily cross into a
# second observation line's date/name/value, producing a garbled answer.
OBSERVATION_LINE_PATTERN = re.compile(
    r"^(\d{4}-\d{2}-\d{2})\s*:\s*(.+?)\s{2,}(\S.*)$"
)
# PROCEDURES and MEDICATIONS both use "<name> (type) for <condition>" --
# only keep the name, same reasoning for both.
HAS_FOR_CLAUSE = {"PROCEDURES", "MEDICATIONS"}


def _strip_type_tag(text: str) -> str:
    """Remove a trailing ' (type)' parenthetical, e.g. 'Gingivitis
    (disorder)' -> 'Gingivitis'. Matches the original 14 gold-set
    entries' style of clean, tag-free answers."""
    return TRAILING_TYPE_TAG.sub("", text).strip()


def generate_from_simple_section(section: str, text: str) -> tuple[str, str] | None:
    """For sections whose dated lines look like 'date : description
    (type)' (CONDITIONS, PROCEDURES, MEDICATIONS, IMMUNIZATIONS,
    IMAGING STUDIES, CARE PLANS, REPORTS): find the first matching line
    (for CONDITIONS specifically, the first line tagged '(disorder)' --
    Synthea also tags unrelated '(finding)'/'(situation)' entries like
    "Received higher education" under CONDITIONS, which don't read as a
    diagnosis), strip the trailing type tag, and (for PROCEDURES/
    MEDICATIONS) drop the ' for <condition>' purpose clause so the
    answer is just the procedure/medication name. Returns (query,
    reference_answer), or None if no matching line is found."""
    if section == "CONDITIONS":
        lines = [
            line for line in text.splitlines() if "(disorder)" in line
        ]
    else:
        lines = text.splitlines()

    match = None
    for line in lines:
        match = DATE_LINE_PATTERN.search(line)
        if match:
            break
    if match is None:
        return None
    date, description = match.group(1), match.group(2).strip()

    if section in HAS_FOR_CLAUSE:
        description = re.split(r"\s+for\s+", description, maxsplit=1)[0]

    answer = _strip_type_tag(description)
    query = SIMPLE_SECTIONS[section].format(date=date)
    return query, answer


def generate_from_observations(text: str) -> tuple[str, str] | None:
    """OBSERVATIONS lines look like 'date : <test name>   <value> <unit>'
    (no type tag, a multi-space gap instead separates the name from the
    value). Checks each line individually (not the whole multi-line
    chunk at once, which previously let the value capture greedily cross
    into a second observation's date/name/value and produce a garbled
    answer) and returns the first line's date, test name, and
    value+unit. Returns (query, reference_answer), or None if no line
    matches."""
    for line in text.splitlines():
        match = OBSERVATION_LINE_PATTERN.match(line.strip())
        if match:
            date, name, value = match.group(1), match.group(2).strip(), match.group(3).strip()
            query = f"What was this patient's {name} recorded on {date}?"
            return query, _strip_type_tag(value)
    return None


def generate_from_allergies(text: str) -> tuple[str, str] | None:
    """ALLERGIES chunks have no date -- just an 'ALLERGIES:' header
    followed by either 'No Known Allergies' or a list of real allergies.
    Returns (query, reference_answer) using the text after the header."""
    body = text.split("ALLERGIES:", maxsplit=1)[-1].strip()
    if not body:
        return None
    return "Does this patient have any known allergies?", body


def generate_query_answer(section: str, text: str) -> tuple[str, str] | None:
    """Dispatch to the right generator for this chunk's section.
    Returns (query, reference_answer), or None if the chunk's text
    didn't match the expected pattern for its section (skipped rather
    than guessed at)."""
    if section in SIMPLE_SECTIONS:
        return generate_from_simple_section(section, text)
    if section == "OBSERVATIONS":
        return generate_from_observations(text)
    if section == "ALLERGIES":
        return generate_from_allergies(text)
    return None


def sample_chunks(n: int, restricted_fraction: float = 0.3) -> list[dict]:
    """Pull n chunks from Postgres, stratified by clearance_tier
    (restricted_fraction from 'restricted', the rest from 'internal'),
    limited to sections generate_query_answer knows how to handle
    (excludes DEMOGRAPHICS and, for 'internal' only, ENCOUNTER_META --
    kept for 'restricted' since an escalated finding can land there).
    Returns a list of {chunk_id, section, clearance_tier, text} dicts."""
    handled_sections = list(SIMPLE_SECTIONS) + ["OBSERVATIONS", "ALLERGIES"]
    n_restricted = round(n * restricted_fraction)
    n_internal = n - n_restricted

    conn = get_connection()
    chunks = []
    for tier, count in [("restricted", n_restricted), ("internal", n_internal)]:
        sections = (
            handled_sections
            if tier == "restricted"
            else [s for s in handled_sections if s != "ENCOUNTER_META"]
        )
        cur = conn.execute(
            "SELECT chunk_id, section, clearance_tier, text "
            "FROM chunks WHERE clearance_tier = %s AND section = ANY(%s) "
            "ORDER BY random() LIMIT %s",
            (tier, sections, count * 2),  # oversample; some will fail to parse
        )
        for chunk_id, section, clearance_tier, text in cur.fetchall():
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "section": section,
                    "clearance_tier": clearance_tier,
                    "text": text,
                }
            )
    conn.close()
    return chunks


def build_candidate_set(n_needed: int, restricted_fraction: float = 0.3) -> list[dict]:
    """Sample chunks (oversampling to absorb parse failures), generate a
    query/answer pair for each via generate_query_answer, and return
    n_needed successes shaped like a gold_set.json entry (query,
    relevant_chunk_ids, reference_answer) -- enforcing restricted_fraction
    as a real per-tier cap, not just an overall count. A flat "stop once
    we have n_needed total" loop would let whichever tier happens to
    parse more easily (restricted, as it turned out) crowd out the other
    tier's quota entirely, since nothing was tracking tier-by-tier
    progress independently. Prints progress every BATCH_SIZE accepted
    entries."""
    n_restricted_target = round(n_needed * restricted_fraction)
    n_internal_target = n_needed - n_restricted_target
    targets = {"restricted": n_restricted_target, "internal": n_internal_target}

    chunks = sample_chunks(n_needed, restricted_fraction)
    accepted = []
    accepted_by_tier = {"restricted": 0, "internal": 0}
    skipped = 0

    for chunk in chunks:
        tier = chunk["clearance_tier"]
        if accepted_by_tier[tier] >= targets[tier]:
            continue  # this tier's quota is already full

        result = generate_query_answer(chunk["section"], chunk["text"])
        if result is None:
            skipped += 1
            continue
        query, reference_answer = result
        accepted.append(
            {
                "query": query,
                "relevant_chunk_ids": [chunk["chunk_id"]],
                "reference_answer": reference_answer,
            }
        )
        accepted_by_tier[tier] += 1
        if len(accepted) % BATCH_SIZE == 0:
            print(f"  ...{len(accepted)}/{n_needed} generated")

        if all(accepted_by_tier[t] >= targets[t] for t in targets):
            break

    print(
        f"Generated {len(accepted)} "
        f"(restricted={accepted_by_tier['restricted']}/{targets['restricted']}, "
        f"internal={accepted_by_tier['internal']}/{targets['internal']}), "
        f"skipped {skipped} (no parseable dated line)"
    )
    return accepted


def main() -> None:
    with open(GOLD_SET_PATH) as f:
        gold_set = json.load(f)

    existing_queries = gold_set["queries"]
    n_needed = TARGET_TOTAL - len(existing_queries)
    next_id = len(existing_queries) + 1

    new_entries = build_candidate_set(n_needed)

    for entry in new_entries:
        entry["query_id"] = f"q{next_id:03d}"
        next_id += 1

    gold_set["queries"] = existing_queries + new_entries

    with open(GOLD_SET_PATH, "w") as f:
        json.dump(gold_set, f, indent=2)

    print(f"Gold set expanded: {len(existing_queries)} -> {len(gold_set['queries'])} queries")


if __name__ == "__main__":
    main()
