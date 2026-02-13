"""Document Pipeline - End-to-end processing"""

from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from .pdf_loader import PDFLoader
from .chunker import DocumentChunker


class DocumentPipeline:
    def __init__(self, chunk_size=512, chunk_overlap=50):
        self.loader = PDFLoader()
        self.chunker = DocumentChunker(chunk_size, chunk_overlap)
        self.stats = {'processed': 0, 'failed': 0, 'pages': 0, 'chunks': 0}
    
    def process_file(self, filepath: str) -> Dict[str, Any]:
        """Process single PDF"""
        path = Path(filepath)
        
        try:
            pages = self.loader.load(filepath)
            chunks = self.chunker.chunk_document(pages)
            
            self.stats['pages'] += len(pages)
            self.stats['chunks'] += len(chunks)
            self.stats['processed'] += 1
            
            return {
                'success': True,
                'source': path.name,
                'pages': len(pages),
                'chunks': chunks,
                'error': None
            }
        except Exception as e:
            self.stats['failed'] += 1
            return {
                'success': False,
                'source': path.name,
                'pages': 0,
                'chunks': [],
                'error': str(e)
            }
    
    def process_batch(self, filepaths: List[str], show_progress=True):
        """Process multiple PDFs"""
        results = []
        
        iterator = tqdm(filepaths) if show_progress else filepaths
        
        for filepath in iterator:
            result = self.process_file(filepath)
            results.append(result)
        
        return results
    
    def get_stats(self):
        return self.stats.copy()


# CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process PDFs to chunks")
    parser.add_argument("files", nargs="+", help="PDF files")
    parser.add_argument("-o", "--output", default="data/processed/chunks.json")
    
    args = parser.parse_args()
    
    pipeline = DocumentPipeline()
    results = pipeline.process_batch(args.files)
    
    # Summary
    stats = pipeline.get_stats()
    print(f"\nProcessed: {stats['processed']}")
    print(f"Failed: {stats['failed']}")
    print(f"Total chunks: {stats['chunks']}")
    
    # Save
    import json
    all_chunks = []
    for r in results:
        if r['success']:
            all_chunks.extend(r['chunks'])
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_chunks, f, indent=2)
    
    print(f"Saved to: {args.output}")