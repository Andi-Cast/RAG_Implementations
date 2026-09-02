import re

SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
PHONE_PATTERN = re.compile(r"\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

STRUCTURED_PATTERNS = [
    (SSN_PATTERN, "ssn"),
    (PHONE_PATTERN, "phone"),
    (DATE_PATTERN, "date"),
]

# Naive heuristic: two-or-more consecutive Title-Case-ish words. Deliberately
# fragile -- this is the illustration of why regex struggles with
# unstructured PII (expect false positives on facility/drug names, and
# false negatives on names that don't look like this).
NAME_CANDIDATE_PATTERN = re.compile(r"\b[A-Z][a-zA-Z]*\s+[A-Z][a-zA-Z]*\b")


def detect_structured_pii(text: str) -> list[tuple[int, int, str]]:
    """Regex-based detection of structured PII: SSNs, phone numbers, dates."""
    spans = []
    for pattern, pii_type in STRUCTURED_PATTERNS:
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end(), pii_type))
    return spans


def detect_name_candidates(text: str) -> list[tuple[int, int, str]]:
    """Naive Title-Case heuristic for names."""
    return [
        (match.start(), match.end(), "name_candidate")
        for match in NAME_CANDIDATE_PATTERN.finditer(text)
    ]


def detect_pii(text: str) -> list[tuple[int, int, str]]:
    """Combined structured-pattern + heuristic-name detection."""
    return detect_structured_pii(text) + detect_name_candidates(text)
