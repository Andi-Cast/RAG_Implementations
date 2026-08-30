from rag.models import Chunk

SECTION_DEFAULT_TIER: dict[str, str] = {
    "DEMOGRAPHICS": "internal",
    "ALLERGIES": "internal",
    "ENCOUNTER_META": "internal",
    "MEDICATIONS": "internal",
    "CONDITIONS": "internal",
    "REPORTS": "internal",
    "OBSERVATIONS": "internal",
    "PROCEDURES": "internal",
}

SENSITIVE_KEYWORDS = [
    "phq-9", "audit-c", "prapare", "abuse",
    "criminal record", "substance use", "psychiatric", "suicide",
]

def tag_clearance_tier(chunk: Chunk) -> str:
    chunk.clearance_tier = SECTION_DEFAULT_TIER.get(chunk.section, "restricted")

    text = chunk.text.lower()
    for keyword in SENSITIVE_KEYWORDS:
        if keyword in text:
            chunk.clearance_tier = "restricted"
    
    return chunk.clearance_tier


        

    