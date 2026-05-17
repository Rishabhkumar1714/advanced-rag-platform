#!/usr/bin/env python3
"""
CLI script to bulk-ingest documents from a directory into the RAG platform.

Usage:
    python scripts/ingest_documents.py --dir ./data/raw --collection my-collection
    python scripts/ingest_documents.py --file ./doc.pdf
    python scripts/ingest_documents.py --url http://localhost:8000
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))


async def ingest_directory(
    directory: str,
    collection_id: str | None = None,
    strategy: str = "recursive",
    api_url: str | None = None,
) -> None:
    """Ingest all supported documents in a directory."""
    from app.services.ingestion_service import IngestionService
    from app.services.document_service import get_document_service

    svc = IngestionService(chunking_strategy=strategy)
    doc_svc = get_document_service()

    path = Path(directory)
    if not path.exists():
        print(f"[ERROR] Directory not found: {directory}")
        sys.exit(1)

    supported = {".pdf", ".docx", ".txt", ".md", ".html"}
    files = [f for f in path.rglob("*") if f.suffix.lower() in supported]

    if not files:
        print(f"[WARN] No supported files found in {directory}")
        return

    print(f"Found {len(files)} file(s) to ingest...")

    success, failed = 0, 0
    for file_path in files:
        try:
            print(f"  Ingesting: {file_path.name} ...", end=" ", flush=True)
            content = file_path.read_bytes()
            doc = await svc.ingest_file(
                file_content=content,
                filename=file_path.name,
                collection_id=collection_id,
            )
            doc_svc.register(doc)
            print(f"✓ ({len(doc.chunks)} chunks)")
            success += 1
        except Exception as exc:
            print(f"✗ FAILED: {exc}")
            failed += 1

    print(f"\nIngestion complete: {success} succeeded, {failed} failed")


async def ingest_single_file(
    file_path: str,
    collection_id: str | None = None,
    strategy: str = "recursive",
) -> None:
    """Ingest a single file."""
    from app.services.ingestion_service import IngestionService
    from app.services.document_service import get_document_service

    svc = IngestionService(chunking_strategy=strategy)
    doc_svc = get_document_service()

    path = Path(file_path)
    if not path.exists():
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)

    print(f"Ingesting: {path.name} ...")
    content = path.read_bytes()
    doc = await svc.ingest_file(
        file_content=content,
        filename=path.name,
        collection_id=collection_id,
    )
    doc_svc.register(doc)
    print(f"✓ Success! Document ID: {doc.document_id}, Chunks: {len(doc.chunks)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bulk-ingest documents into the Advanced RAG Platform"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", help="Directory of documents to ingest")
    group.add_argument("--file", help="Single file to ingest")

    parser.add_argument("--collection", help="Collection ID to assign documents to")
    parser.add_argument(
        "--strategy",
        choices=["recursive", "semantic", "sliding_window"],
        default="recursive",
        help="Chunking strategy (default: recursive)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.dir:
        asyncio.run(
            ingest_directory(
                directory=args.dir,
                collection_id=args.collection,
                strategy=args.strategy,
            )
        )
    else:
        asyncio.run(
            ingest_single_file(
                file_path=args.file,
                collection_id=args.collection,
                strategy=args.strategy,
            )
        )
