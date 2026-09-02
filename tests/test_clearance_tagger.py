from rag.ingest.clearance_tagger import tag_clearance_tier
from rag.models import Chunk


def make_chunk(text: str, section: str) -> Chunk:
    return Chunk(text=text, section=section, patient_id="p1", source_file="f.txt")


def test_section_default_no_keyword_hit():
    chunk = make_chunk(
        text="2007-07-08 : lisinopril 10 MG Oral Tablet for Essential hypertension",
        section="MEDICATIONS",
    )
    assert tag_clearance_tier(chunk) == "internal"


def test_unrecognized_section_fails_closed():
    chunk = make_chunk(text="some content", section="UNKNOWN_SECTION")
    assert tag_clearance_tier(chunk) == "restricted"


def test_abuse_keyword_escalates_conditions_chunk():
    chunk = make_chunk(
        text="2007-07-08 : Victim of intimate partner abuse (finding)",
        section="CONDITIONS",
    )
    assert tag_clearance_tier(chunk) == "restricted"


def test_keyword_escalation_is_case_insensitive():
    chunk = make_chunk(
        text="2007-07-08 : Victim of intimate partner ABUSE (finding)",
        section="CONDITIONS",
    )
    assert tag_clearance_tier(chunk) == "restricted"


def test_phq9_keyword_escalates_observations_chunk():
    chunk = make_chunk(
        text="2007-07-08 : PHQ-9 quick depression assessment panel [Reported.PHQ]",
        section="OBSERVATIONS",
    )
    assert tag_clearance_tier(chunk) == "restricted"


def test_no_keyword_hit_keeps_section_default_for_observations():
    chunk = make_chunk(
        text="2007-07-08 : Heart rate 61.0 /min",
        section="OBSERVATIONS",
    )
    assert tag_clearance_tier(chunk) == "internal"


def test_routine_immunizations_are_not_restricted():
    chunk = make_chunk(
        text="2019-02-21 : Hep A, ped/adol, 2 dose\n  2019-02-21 : MMR",
        section="IMMUNIZATIONS",
    )
    assert tag_clearance_tier(chunk) == "internal"


def test_routine_imaging_studies_are_not_restricted():
    chunk = make_chunk(
        text="2026-07-25 : Ultrasound, Heart structure (body structure)",
        section="IMAGING STUDIES",
    )
    assert tag_clearance_tier(chunk) == "internal"


def test_routine_care_plans_are_not_restricted():
    chunk = make_chunk(
        text="1988-04-17 : Diabetes self management plan (record artifact)",
        section="CARE PLANS",
    )
    assert tag_clearance_tier(chunk) == "internal"
