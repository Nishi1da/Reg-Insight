#!/usr/bin/env python
"""Index Documents - CLI for full ingestion pipeline"""

import sys
from pathlib import Path
import argparse
import json
import time

sys.path.insert(0, str(Path(__file__).parent))

from src.embeddings.ingestion_pipeline import BatchIngestionPipeline


def main():
    parser = argparse.ArgumentParser(description="Index PDFs to vector store")
    parser.add_argument("path", help="PDF file or directory to index")
    parser.add_argument("-c", "--collection", default="regulations", help="Collection name")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Chunk overlap")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--stats", action="store_true", help="Show detailed stats")
    
    args = parser.parse_args()
    
    target_path = Path(args.path)
    
    # Collect PDFs
    if target_path.is_dir():
        pdf_files = list(target_path.glob("*.pdf"))
    else:
        pdf_files = [target_path] if target_path.suffix == '.pdf' else []
    
    if not pdf_files:
        print(f" No PDFs found: {args.path}")
        return 1
    
    print(f" Found {len(pdf_files)} PDF(s) to index")
    print(f" Collection: {args.collection}")
    print(f"  Chunk size: {args.chunk_size}, Overlap: {args.chunk_overlap}")
    print("-" * 60)
    
    # Run pipeline
    start_time = time.time()
    
    pipeline = BatchIngestionPipeline(batch_size=args.batch_size)
    stats = pipeline.ingest_from_pipeline(
        [str(f) for f in pdf_files],
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    elapsed = time.time() - start_time
    
    print("-" * 60)
    print(f" Indexing complete in {elapsed:.1f}s")
    print(f"   Chunks processed: {stats['chunks_processed']}")
    print(f"   Failed: {stats['chunks_failed']}")
    print(f"   Speed: {stats.get('chunks_per_second', 0):.1f} chunks/sec")
    
    # Stats
    if args.stats:
        print("\n Detailed stats:")
        print(json.dumps(stats, indent=2))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())