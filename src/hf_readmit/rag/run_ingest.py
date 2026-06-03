from __future__ import annotations

import argparse
import time
from pathlib import Path

from hf_readmit.config import settings
from hf_readmit.rag.bm25_index import BM25Index
from hf_readmit.rag.ingest import build_guideline_corpus
from hf_readmit.rag.retriever import ChromaVectorStore, OpenAIEmbeddings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest heart failure guideline PDFs into BM25 and Chroma retrieval stores."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("guidelines"),
        help="Directory containing guideline PDF source documents.",
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=Path("chroma_db"),
        help="Local ChromaDB persistence directory.",
    )
    parser.add_argument(
        "--bm25-path",
        type=Path,
        default=Path("models/bm25_index.pkl"),
        help="Local path for the BM25 index pickle file.",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="hf_guidelines",
        help="ChromaDB collection name.",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API key for embeddings; falls back to environment settings if blank.",
    )
    args = parser.parse_args()
    start_time = time.perf_counter()

    corpus = build_guideline_corpus(args.source_dir)
    if not corpus:
        raise SystemExit(f"No PDF guideline documents found in {args.source_dir}")

    counts: dict[str, int] = {}
    for chunk in corpus:
        counts[chunk.source_name] = counts.get(chunk.source_name, 0) + 1

    bm25_index = BM25Index(corpus)
    bm25_index.save(args.bm25_path)

    openai_key = args.openai_api_key or settings.openai_api_key
    embedder = OpenAIEmbeddings(api_key=openai_key) if openai_key else None
    chroma_store = ChromaVectorStore(persist_path=args.chroma_path, collection_name=args.collection_name)
    chroma_store.add_documents(corpus, embedder=embedder)

    total_chunks = len(corpus)
    elapsed = time.perf_counter() - start_time
    for source_name, count in counts.items():
        print(f"{source_name}: {count} chunks")
    print(f"Total chunks: {total_chunks}")
    print(f"Time taken: {elapsed:.2f}s")
    print(f"Built BM25 index with {total_chunks} chunks and saved to {args.bm25_path}")
    print(f"Persisted ChromaDB collection '{args.collection_name}' to {args.chroma_path}")


if __name__ == "__main__":
    main()
