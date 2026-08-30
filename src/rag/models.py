from dataclasses import dataclass

CLEARANCE_TIERS: dict[str, int] = {
    "public" : 0,
    "internal" : 1,
    "confidential" : 2,
    "restricted" : 3,
}

@dataclass
class Chunk:
    text: str
    section: str
    patient_id: str
    source_file: str
    clearance_tier: str = ""
