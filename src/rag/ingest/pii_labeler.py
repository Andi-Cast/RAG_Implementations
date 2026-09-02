import re

from rag.models import Chunk


def extract_patient_identifiers(demographics_chunk: Chunk) -> dict[str, str]:
    """Pull known identifying values (name, birth_date) out of a
    DEMOGRAPHICS chunk's text."""

    lines = demographics_chunk.text.splitlines()
    identifiers = {"name": lines[0].strip()}

    for line in lines:
        if line.strip().startswith("Birth Date:"):
            identifiers["birth_date"] = line.split(":", 1)[1].strip()
            break

    return identifiers


def label_pii_spans(chunk: Chunk, identifiers: dict[str, str]) -> list[tuple[int, int, str]]:
    """Find every occurrence of each known identifier value in chunk.text.
    Returns (start, end, pii_type) character-offset spans."""

    spans = []
    for pii_type, value in identifiers.items():
        if not value:
            continue
        for match in re.finditer(re.escape(value), chunk.text):
            spans.append((match.start(), match.end(), pii_type))

    return spans
