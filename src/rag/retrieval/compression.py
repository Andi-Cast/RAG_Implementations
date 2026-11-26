import numpy as np
from langfuse import observe

from rag.observability.tracing import get_langfuse, mask_chunk_ids
from rag.retrieval.dense import get_model
from rag.retrieval.rerank import get_chunk_texts


def compress_chunk(query: str, chunk_text: str, top_n: int = 2) -> str:
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


@observe(name="compress-chunks", as_type="tool", capture_input=False, capture_output=False)
def compress_chunks(query: str, chunk_ids: list[str], top_n: int = 2) -> dict[str, str]:
    texts = get_chunk_texts(chunk_ids)
    compressed = {
        chunk_id: compress_chunk(query, text, top_n)
        for chunk_id, text in texts.items()
    }

    get_langfuse().update_current_span(
        input={"chunk_ids": mask_chunk_ids(chunk_ids), "top_n": top_n},
        output={"chunk_ids": mask_chunk_ids(list(compressed.keys()))},
    )

    return compressed
