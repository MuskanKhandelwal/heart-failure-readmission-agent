from __future__ import annotations

from typing import Iterable


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    retrieved_at_k = retrieved[:k]
    if not retrieved_at_k:
        return 0.0
    return sum(1 for item in retrieved_at_k if item in relevant) / len(retrieved_at_k)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for item in retrieved[:k] if item in relevant) / len(relevant)


def mean_reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def _matched_source(hit: dict, relevant: set[str]) -> str:
    """Resolve a hit to the relevant source it matches, for source-level eval.

    Chunk IDs look like ``aha_hf_guideline_2022_p42_0`` while relevant sources
    look like ``aha_hf_guideline_2022``. A hit matches a relevant source if the
    chunk's ``source_name`` metadata equals it (with or without a ``.pdf``
    extension) OR the chunk_id starts with it. When nothing matches, the raw
    chunk_id is returned so it counts toward precision but not as a relevant hit.
    """
    metadata = hit.get("metadata") or {}
    source = str(metadata.get("source_name", ""))
    source_stem = source[:-4] if source.endswith(".pdf") else source
    chunk_id = str(hit.get("chunk_id", ""))
    for rel in relevant:
        if source == rel or source_stem == rel or chunk_id.startswith(rel):
            return rel
    return chunk_id or source


def evaluate_retrieval(
    queries: Iterable[tuple[str, set[str]]],
    retriever,
    top_k: int = 5,
) -> dict[str, float]:
    precisions = []
    recalls = []
    mrrs = []

    for query_text, relevant_ids in queries:
        hits = retriever.retrieve(query_text, top_k=top_k)
        # Map each hit to a source-level id, then dedupe (preserving order) so a
        # source matched by several chunks counts once in source-level metrics.
        seen: set[str] = set()
        retrieved_ids: list[str] = []
        for hit in hits:
            source_id = _matched_source(hit, relevant_ids)
            if source_id not in seen:
                seen.add(source_id)
                retrieved_ids.append(source_id)
        precisions.append(precision_at_k(retrieved_ids, relevant_ids, top_k))
        recalls.append(recall_at_k(retrieved_ids, relevant_ids, top_k))
        mrrs.append(mean_reciprocal_rank(retrieved_ids, relevant_ids))

    return {
        "precision_at_k": sum(precisions) / len(precisions) if precisions else 0.0,
        "recall_at_k": sum(recalls) / len(recalls) if recalls else 0.0,
        "mean_reciprocal_rank": sum(mrrs) / len(mrrs) if mrrs else 0.0,
        "queries_evaluated": len(queries),
    }
