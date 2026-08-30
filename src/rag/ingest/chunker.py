import re
from pathlib import Path
from rag.models import Chunk

SECTION_SEPARATOR = re.compile(r'^-{10,}$', re.MULTILINE)
SUBSECTION_HEADER = re.compile(r'^\s*([A-Z][A-Z ]+):\s*$', re.MULTILINE)

def split_top_level_sections(text: str) -> list[str]:
    """Split the raw file on the dashed separator lines. Returns blocks in
    file order: [demographics+header, allergies, encounter, continuing]."""

    raw_blocks = SECTION_SEPARATOR.split(text)
    return [block.strip() for block in raw_blocks if block.strip()]

def split_encounter_subsections(encounter_block: str) -> list[tuple[str, str]]:
    """Within the ENCOUNTER block, split on ALL-CAPS colon-terminated
    headers (e.g. 'MEDICATIONS:'). Returns (section_name, section_text) pairs."""

    parts = SUBSECTION_HEADER.split(encounter_block)

    sections = []

    meta_text = parts[0].strip()
    if meta_text:
        sections.append(("ENCOUNTER_META", meta_text))

    for i in range(1, len(parts), 2):
        section_name = parts[i].strip()
        section_text = parts[i + 1].strip()
        if section_text:
            sections.append((section_name, section_text))

    return sections

def chunk_encounter_file(file_path: str) -> list[Chunk]:
    """Read one Synthea text_encounters file, extract patient_id from the
    filename, and return one Chunk per sub-section (demographics, allergies,
    and each ENCOUNTER sub-section). Skips CONTINUING."""

    text = Path(file_path).read_text()
    patient_id = Path(file_path).stem

    top_sections = split_top_level_sections(text)
    top_labels = ["DEMOGRAPHICS", "ALLERGIES", "ENCOUNTER", "CONTINUING"]

    chunks = []
    for label, block in zip(top_labels, top_sections):
        if label == "CONTINUING":
            continue

        if label == "ENCOUNTER":
            for section_name, section_text in split_encounter_subsections(block):
                chunks.append(Chunk(
                    text=section_text,
                    section=section_name,
                    patient_id=patient_id,
                    source_file=file_path,
                ))
        else:
            chunks.append(Chunk(
                text=block,
                section=label,
                patient_id=patient_id,
                source_file=file_path,
            ))

    return chunks

