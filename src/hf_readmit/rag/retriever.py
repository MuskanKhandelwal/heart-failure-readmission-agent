from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from chromadb import PersistentClient
from chromadb.config import Settings

from hf_readmit.config import settings
from hf_readmit.rag.bm25_index import BM25Index
from hf_readmit.rag.ingest import GuidelineChunk


class OpenAIEmbeddings:
    """OpenAI embedding helper for semantic retrieval."""

    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-large") -> None:
        self.api_key = api_key or settings.openai_api_key
        self.model = model

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The openai package is required for semantic embeddings. "
                "Install it with `pip install openai`.") from exc

        self.client = OpenAI(api_key=self.api_key)

    # OpenAI enforces a per-request cap of 300k tokens; stay safely under it.
    _MAX_TOKENS_PER_REQUEST = 250_000

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for batch in self._batched(texts):
            response = self.client.embeddings.create(model=self.model, input=batch)
            embeddings.extend(item.embedding for item in response.data)
        return embeddings

    def _batched(self, texts: list[str]) -> Iterable[list[str]]:
        """Yield batches of texts kept under the per-request token budget.

        Tokens are estimated as ~1 token per 4 characters, which is a
        conservative heuristic for English text.
        """
        batch: list[str] = []
        batch_tokens = 0
        for text in texts:
            est_tokens = max(1, len(text) // 4)
            if batch and batch_tokens + est_tokens > self._MAX_TOKENS_PER_REQUEST:
                yield batch
                batch = []
                batch_tokens = 0
            batch.append(text)
            batch_tokens += est_tokens
        if batch:
            yield batch


class ChromaVectorStore:
    """Wrapper around a persistent ChromaDB collection for guideline chunks."""

    def __init__(self, persist_path: Path, collection_name: str = "hf_guidelines") -> None:
        self.persist_path = persist_path
        self.client = PersistentClient(path=str(persist_path), settings=Settings())
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(
        self,
        chunks: list[GuidelineChunk],
        embedder: Optional[OpenAIEmbeddings] = None,
    ) -> None:
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.to_metadata() for chunk in chunks]

        if embedder is not None:
            embeddings = embedder.embed(documents)
        else:
            embeddings = None

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def query(self, query_text: str, embedder: Optional[OpenAIEmbeddings] = None, top_k: int = 5) -> list[dict[str, object]]:
        if embedder is not None:
            embeddings = embedder.embed([query_text])
            query_embeddings = [embeddings[0]]
            result = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        else:
            result = self.collection.query(
                query_texts=[query_text],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

        ids = result.get("ids", [])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])
        distances = result.get("distances", [])

        if ids and isinstance(ids[0], list):
            ids = ids[0]
            documents = documents[0] if documents else []
            metadatas = metadatas[0] if metadatas else []
            distances = distances[0] if distances else []

        hits: list[dict[str, object]] = []
        for index, chunk_id in enumerate(ids):
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "document": documents[index] if index < len(documents) else "",
                    "metadata": metadatas[index] if index < len(metadatas) else {},
                    "distance": float(distances[index]) if index < len(distances) and distances[index] is not None else None,
                }
            )
        return hits


class HybridRetriever:
    """Hybrid retriever combining BM25 and semantic ranking."""

    def __init__(
        self,
        bm25_index: BM25Index,
        chroma_store: Optional[ChromaVectorStore] = None,
        embedder: Optional[OpenAIEmbeddings] = None,
    ) -> None:
        self.bm25_index = bm25_index
        self.chroma_store = chroma_store
        self.embedder = embedder

    def retrieve(self, query_text: str, top_k: int = 5) -> list[dict[str, object]]:
        bm25_hits = self.bm25_index.query(query_text, top_k=top_k)
        semantic_hits: list[dict[str, object]] = []

        if self.chroma_store is not None:
            semantic_hits = self.chroma_store.query(query_text, embedder=self.embedder, top_k=top_k)

        merged: dict[str, dict[str, object]] = {}

        for rank, hit in enumerate(bm25_hits):
            merged[hit.chunk_id] = {
                "chunk_id": hit.chunk_id,
                "document": hit.document,
                "metadata": hit.metadata,
                "bm25_score": hit.score,
                "semantic_score": None,
                "combined_score": float(hit.score),
                "source": "bm25",
            }

        for rank, hit in enumerate(semantic_hits):
            semantic_score = 1.0 / (1.0 + hit.get("distance", 0.0)) if hit.get("distance") is not None else 1.0 / (rank + 1)
            if hit["chunk_id"] in merged:
                existing = merged[hit["chunk_id"]]
                existing["semantic_score"] = semantic_score
                existing["combined_score"] = existing["combined_score"] + 0.5 * semantic_score
                existing["source"] = "hybrid"
            else:
                merged[hit["chunk_id"]] = {
                    "chunk_id": hit["chunk_id"],
                    "document": hit["document"],
                    "metadata": hit["metadata"],
                    "bm25_score": None,
                    "semantic_score": semantic_score,
                    "combined_score": 0.5 * semantic_score,
                    "source": "semantic",
                }

        results = sorted(merged.values(), key=lambda item: item["combined_score"], reverse=True)
        return results[:top_k]
