import re

from langfuse import observe
from rag.observability.tracing import get_langfuse

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
    return [label for pattern, label in INJECTION_PATTERNS if pattern.search(text)]


@observe(name="sanitize-context", as_type="tool", capture_input=False, capture_output=False)
def sanitize_retrieved_context(chunks: list[str]) -> list[str]:
    sanitized = []
    all_labels = set()
    for chunk in chunks:
        cleaned = chunk
        for pattern, label in INJECTION_PATTERNS:
            if pattern.search(cleaned):
                all_labels.add(label)
            cleaned = pattern.sub(f"[REDACTED-{label.upper()}]", cleaned)
        sanitized.append(cleaned)

    get_langfuse().update_current_span(
        input={"num_chunks": len(chunks)},
        output={"num_chunks": len(sanitized), "patterns_found": sorted(all_labels)},
    )

    return sanitized
