import numpy as np

from rag.retrieval.dense import get_model
from rag.retrieval.rerank import get_chunk_texts


def compress_chunk(query: str, chunk_text: str, top_n: int = 2) -> str:
    """Keep only the top_n most query-relevant lines from chunk_text, in
    their original order, dropping the rest. Splits on lines rather than
    sentences since this corpus's chunks are line-structured facts
    ("date : finding") rather than prose."""
    lines = [line for line in chunk_text.splitlines() if line.strip()]
    if len(lines) <= top_n:
        return chunk_text

    model = get_model()
    query_embedding = model.encode(query, normalize_embeddings=True)
    line_embeddings = model.encode(lines, normalize_embeddings=True)

    similarities = line_embeddings @ query_embedding
    top_indices = sorted(range(len(lines)), key=lambda i: similarities[i], reverse=True)[:top_n]
    kept_in_order = sorted(top_indices)

    return "\n".join(lines[i] for i in kept_in_order)


def compress_chunks(query: str, chunk_ids: list[str], top_n: int = 2) -> dict[str, str]:
    """Fetch each chunk's text and compress it against the query. Returns
    {chunk_id: compressed_text}, ready to hand to a generation step."""
    texts = get_chunk_texts(chunk_ids)
    return {
        chunk_id: compress_chunk(query, text, top_n)
        for chunk_id, text in texts.items()
    }
