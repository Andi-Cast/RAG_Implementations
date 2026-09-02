from rag.security.pii_redaction import (
    detect_name_candidates,
    detect_pii,
    detect_structured_pii,
)


def test_detect_structured_pii_finds_date():
    text = "2007-07-08 : Encounter at BETH ISRAEL DEACONESS HOSPITAL"
    spans = detect_structured_pii(text)

    assert len(spans) == 1
    start, end, pii_type = spans[0]
    assert pii_type == "date"
    assert text[start:end] == "2007-07-08"


def test_detect_structured_pii_finds_ssn():
    text = "SSN on file: 123-45-6789"
    spans = detect_structured_pii(text)

    assert ("ssn" in [t for _, _, t in spans])
    start, end, pii_type = next(s for s in spans if s[2] == "ssn")
    assert text[start:end] == "123-45-6789"


def test_detect_structured_pii_finds_phone():
    text = "Call back at 555-123-4567 to schedule"
    spans = detect_structured_pii(text)

    assert any(pii_type == "phone" for _, _, pii_type in spans)


def test_detect_structured_pii_no_match_returns_empty():
    assert detect_structured_pii("no identifiers in this text at all") == []


def test_detect_name_candidates_catches_two_word_title_case():
    text = "Beth Israel Deaconess"
    spans = detect_name_candidates(text)

    assert len(spans) >= 1


def test_detect_name_candidates_misses_digit_suffixed_synthetic_names():
    # Known limitation: Synthea names like "Leon728 Hayes766" have digits
    # appended, which breaks the naive [A-Z][a-zA-Z]* pattern. This test
    # documents the false negative rather than asserting correct behavior.
    text = "Patient Leon728 Hayes766 was seen today"
    spans = detect_name_candidates(text)

    assert not any(text[start:end] == "Leon728 Hayes766" for start, end, _ in spans)


def test_detect_name_candidates_false_positive_on_facility_name():
    # Known limitation: all-caps facility names match the same pattern as
    # real names, since the regex allows uppercase letters throughout.
    text = "BETH ISRAEL DEACONESS HOSPITAL PLYMOUTH INC"
    spans = detect_name_candidates(text)

    assert len(spans) > 0


def test_detect_pii_combines_structured_and_name_candidates():
    text = "2007-07-08 : Beth Israel Deaconess"
    spans = detect_pii(text)

    pii_types = {pii_type for _, _, pii_type in spans}
    assert "date" in pii_types
    assert "name_candidate" in pii_types
