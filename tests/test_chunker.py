import textwrap
from pathlib import Path

from rag.ingest.chunker import (
    chunk_encounter_file,
    split_encounter_subsections,
    split_top_level_sections,
)

FIXTURE_FILE = Path(__file__).parent / "fixtures" / "sample_encounter.txt"


def test_split_top_level_sections_basic():
    text = textwrap.dedent("""\
        demographics stuff
        ----------
        ALLERGIES:
        No Known Allergies
        ----------
        ENCOUNTER
        some encounter content
        ----------
        CONTINUING
        history stuff
        ----------
        """)
    blocks = split_top_level_sections(text)

    assert len(blocks) == 4
    assert blocks[0] == "demographics stuff"
    assert blocks[1] == "ALLERGIES:\nNo Known Allergies"
    assert blocks[2] == "ENCOUNTER\nsome encounter content"
    assert blocks[3] == "CONTINUING\nhistory stuff"


def test_split_top_level_sections_drops_empty_blocks():
    text = "----------\n----------\ncontent\n----------\n"
    blocks = split_top_level_sections(text)

    assert blocks == ["content"]


def test_split_encounter_subsections_pairs_headers_with_content():
    encounter_block = """ENCOUNTER
    2007-07-08 : Encounter at BETH ISRAEL DEACONESS HOSPITAL PLYMOUTH INC
    Type: outpatient

    MEDICATIONS:
    2007-07-08 : amLODIPine 2.5 MG Oral Tablet for Essential hypertension (disorder)

    CONDITIONS:
    2007-07-08 : Victim of intimate partner abuse (finding)

    CARE PLANS:

    """
    sections = split_encounter_subsections(encounter_block)
    section_names = [name for name, _ in sections]

    assert section_names == ["ENCOUNTER_META", "MEDICATIONS", "CONDITIONS"]

    meta_text = dict(sections)["ENCOUNTER_META"]
    assert "BETH ISRAEL DEACONESS" in meta_text
    assert "Type: outpatient" in meta_text

    medications_text = dict(sections)["MEDICATIONS"]
    assert "amLODIPine" in medications_text


def test_split_encounter_subsections_drops_empty_sections():
    encounter_block = """ENCOUNTER
    some meta

    IMMUNIZATIONS:

    IMAGING STUDIES:

    """
    sections = split_encounter_subsections(encounter_block)
    section_names = [name for name, _ in sections]

    assert "IMMUNIZATIONS" not in section_names
    assert "IMAGING STUDIES" not in section_names


def test_chunk_encounter_file_returns_expected_sections():
    chunks = chunk_encounter_file(str(FIXTURE_FILE))
    section_names = [c.section for c in chunks]

    assert section_names == [
        "DEMOGRAPHICS",
        "ALLERGIES",
        "ENCOUNTER_META",
        "MEDICATIONS",
        "CONDITIONS",
        "REPORTS",
        "OBSERVATIONS",
        "PROCEDURES",
    ]
    assert "CONTINUING" not in section_names


def test_chunk_encounter_file_sets_patient_id_and_source_file():
    chunks = chunk_encounter_file(str(FIXTURE_FILE))

    for chunk in chunks:
        assert chunk.patient_id == FIXTURE_FILE.stem
        assert chunk.source_file == str(FIXTURE_FILE)


def test_chunk_encounter_file_demographics_chunk_has_expected_content():
    chunks = chunk_encounter_file(str(FIXTURE_FILE))
    demographics = next(c for c in chunks if c.section == "DEMOGRAPHICS")

    assert "Leon728 Hayes766" in demographics.text
    assert "Birth Date" in demographics.text
