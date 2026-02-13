#!/usr/bin/env python
"""Simple CLI to load documents"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion.document_pipeline import DocumentPipeline


def main():
    if len(sys.argv) < 2:
        print("Usage: python load_documents.py <pdf_or_folder>")
        return 1
    
    path = Path(sys.argv[1])
    
    if path.is_dir():
        files = list(path.glob("*.pdf"))
    else:
        files = [path]
    
    if not files:
        print(f"No PDFs found: {path}")
        return 1
    
    pipeline = DocumentPipeline()
    results = pipeline.process_batch([str(f) for f in files])
    
    for r in results:
        status = "✅" if r['success'] else "❌"
        chunks = len(r['chunks']) if r['success'] else 0
        print(f"{status} {r['source']}: {chunks} chunks")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())