from dotenv import load_dotenv

load_dotenv()

from langfuse import get_client

TRACE_TEXT_MAX_LEN = 60


def get_langfuse():
    return get_client()


def mask_text(value: str, max_len: int = TRACE_TEXT_MAX_LEN) -> str:
    if value is None:
        return value
    if len(value) <= max_len:
        return value
    return value[:max_len] + "…"


def mask_chunk_ids(chunk_ids: list[str]) -> dict:
    return {
        "count": len(chunk_ids),
        "sample": [mask_text(c, 24) for c in chunk_ids[:3]],
    }
