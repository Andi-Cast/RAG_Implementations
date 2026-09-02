from rag.ingest.pii_labeler import extract_patient_identifiers, label_pii_spans
from rag.models import Chunk


def make_chunk(text: str, section: str = "DEMOGRAPHICS") -> Chunk:
    return Chunk(text=text, section=section, patient_id="p1", source_file="f.txt")


def test_extract_patient_identifiers_basic():
    chunk = make_chunk(
        "Leon728 Hayes766\n"
        "================\n"
        "Race:                White\n"
        "Ethnicity:           Non-Hispanic\n"
        "Gender:              M\n"
        "Age:                 DECEASED\n"
        "Birth Date:          1948-04-11\n"
        "Marital Status:      M"
    )
    identifiers = extract_patient_identifiers(chunk)

    assert identifiers["name"] == "Leon728 Hayes766"
    assert identifiers["birth_date"] == "1948-04-11"


def test_label_pii_spans_finds_single_occurrence():
    chunk = make_chunk("Patient name: Leon728 Hayes766", section="MEDICATIONS")
    identifiers = {"name": "Leon728 Hayes766"}

    spans = label_pii_spans(chunk, identifiers)

    assert len(spans) == 1
    start, end, pii_type = spans[0]
    assert pii_type == "name"
    assert chunk.text[start:end] == "Leon728 Hayes766"


def test_label_pii_spans_finds_multiple_occurrences():
    chunk = make_chunk("1948-04-11 note, follow up on 1948-04-11", section="OBSERVATIONS")
    identifiers = {"birth_date": "1948-04-11"}

    spans = label_pii_spans(chunk, identifiers)

    assert len(spans) == 2


def test_label_pii_spans_no_match_returns_empty():
    chunk = make_chunk("nothing sensitive here", section="MEDICATIONS")
    identifiers = {"name": "Leon728 Hayes766", "birth_date": "1948-04-11"}

    assert label_pii_spans(chunk, identifiers) == []


def test_label_pii_spans_skips_empty_identifier_values():
    chunk = make_chunk("some text", section="MEDICATIONS")
    identifiers = {"name": "", "birth_date": "1948-04-11"}

    assert label_pii_spans(chunk, identifiers) == []
