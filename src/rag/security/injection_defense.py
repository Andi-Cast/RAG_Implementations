import re

# Naive, regex/keyword-based patterns -- same deliberate tradeoff as
# pii_redaction.py: easy to bypass with rephrasing, but illustrates what a
# hand-rolled defense catches (and sets up a comparison point against
# Bedrock Guardrails later).
IGNORE_INSTRUCTIONS_PATTERN = re.compile(
    r"\b(ignore|disregard|forget)\s+(all\s+)?(the\s+)?(previous|prior|above)\s+instructions\b",
    re.IGNORECASE,
)
ROLE_OVERRIDE_PATTERN = re.compile(
    r"\b(you are now|act as|pretend (to be|you are)|new instructions?\s*:)",
    re.IGNORECASE,
)
FAKE_TURN_MARKER_PATTERN = re.compile(
    r"(^|\n)\s*(system|assistant|user)\s*:|\[INST\]|<\|im_start\|>",
    re.IGNORECASE,
)

INJECTION_PATTERNS = [
    (IGNORE_INSTRUCTIONS_PATTERN, "ignore_instructions"),
    (ROLE_OVERRIDE_PATTERN, "role_override"),
    (FAKE_TURN_MARKER_PATTERN, "fake_turn_marker"),
]


def detect_injection_patterns(text: str) -> list[str]:
    """Scan chunk/query text for known prompt-injection patterns
    (e.g. "ignore previous instructions", role-play override attempts,
    fake system/assistant turn markers) and return the list of matched
    pattern names/strings found. Deliberately naive/heuristic, same as
    pii_redaction.py's approach -- documents what a regex/keyword-based
    defense catches and misses."""
    return [label for pattern, label in INJECTION_PATTERNS if pattern.search(text)]


def sanitize_retrieved_context(chunks: list[str]) -> list[str]:
    """Given retrieved chunk texts that will be inserted into the LLM
    prompt, strip or neutralize any detected injection patterns before
    they reach the model -- the actual defense step, separate from
    detection."""
    sanitized = []
    for chunk in chunks:
        cleaned = chunk
        for pattern, label in INJECTION_PATTERNS:
            cleaned = pattern.sub(f"[REDACTED-{label.upper()}]", cleaned)
        sanitized.append(cleaned)
    return sanitized
