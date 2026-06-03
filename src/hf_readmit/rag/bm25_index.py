from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from hf_readmit.rag.ingest import GuidelineChunk


@dataclass
class BM25Hit:
    chunk_id: str
    score: float
    document: str
    metadata: dict[str, Any]


class BM25Index:
    """A simple BM25 retrieval index over guideline chunks."""

    def __init__(self, chunks: list[GuidelineChunk]) -> None:
        self.chunks = chunks
        self.documents = [chunk.text for chunk in chunks]
        self.metadatas = [chunk.to_metadata() for chunk in chunks]
        self.ids = [chunk.chunk_id for chunk in chunks]
        self.tokenized_documents = [self._tokenize(text) for text in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_documents)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token.lower() for token in re.findall(r"\w+", text) if token]

    def query(self, query_text: str, top_k: int = 5) -> list[BM25Hit]:
        tokens = self._tokenize(query_text)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]
        return [
            BM25Hit(
                chunk_id=self.ids[index],
                score=float(scores[index]),
                document=self.documents[index],
                metadata=self.metadatas[index],
            )
            for index in top_indices
            if scores[index] > 0
        ]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with path.open("rb") as handle:
            loaded = pickle.load(handle)
        if not isinstance(loaded, BM25Index):
            raise ValueError(f"Expected BM25Index at {path}, got {type(loaded)}")
        return loaded
