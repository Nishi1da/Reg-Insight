"""Document Pipeline - End-to-end: PDF → Chunks → ChromaDB"""

from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from .pdf_loader import PDFLoader
from .chunker import DocumentChunker
from ..embeddings.chroma_manager import ChromaManager  


class DocumentPipeline:
    def __init__(self, chunk_size=512, chunk_overlap=50):
        self.loader = PDFLoader()
        self.chunker = DocumentChunker(chunk_size, chunk_overlap)
        self.chroma = ChromaManager()  
        self.stats = {
            'processed': 0, 'failed': 0,
            'pages': 0, 'chunks': 0,
            'added_to_chroma': 0  
        }
    
    def process_file(self, filepath: str) -> Dict[str, Any]:
        """Process single PDF and add to ChromaDB"""
        path = Path(filepath)
        
        try:
            pages = self.loader.load(filepath)
            chunks = self.chunker.chunk_document(pages)
            
            #  ADD: Convert to ChromaDB format and store
            chroma_chunks = self._convert_to_chroma_format(chunks, path.name)

            if "regulations" in str(path).lower():
                self._add_to_chroma(chroma_chunks, "regulations")
            elif "policies" in str(path).lower():
                self._add_to_chroma(chroma_chunks, "policies")
            
            self.stats['pages'] += len(pages)
            self.stats['chunks'] += len(chunks)
            self.stats['added_to_chroma'] += len(chroma_chunks)
            self.stats['processed'] += 1
            
            return {
                'success': True,
                'source': path.name,
                'pages': len(pages),
                'chunks': len(chunks),
                'error': None
            }
        except Exception as e:
            self.stats['failed'] += 1
            return {
                'success': False,
                'source': path.name,
                'pages': 0, 'chunks': 0,
                'error': str(e)
            }
    def _add_to_chroma(self, chunks, collection_name):
        if not chunks:
            return
        
        texts = [c['text'] for c in chunks]
        metadatas = [c['metadata'] for c in chunks]
        ids = [f"{collection_name}_{i}" for i in range(len(chunks))]
        
        self.chroma.add_documents(
        documents=texts,
        embeddings=None,   
        metadatas=metadatas,
        ids=ids,
        collection_name=collection_name
    )    
    
    def _convert_to_chroma_format(self, chunks: List[Dict], filename: str) -> List[Dict]:
        """
         KEY FIX: Convert your chunk format to ChromaDB format
        
        Your chunks have: {'content': '...', 'metadata': {...}}
        ChromaDB needs:  {'text': '...', 'metadata': {...}}
        """
        converted = []
        for i, chunk in enumerate(chunks):
            # Handle both 'content' and 'text' keys
            text = chunk.get('content') or chunk.get('text', '')
            
            converted.append({
                'text': text,
                'metadata': {
                    'source': filename,
                    'chunk_index': i,
                    **chunk.get('metadata', {})  # Preserve existing metadata
                }
            })
        return converted
    
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
    
    #   Check ChromaDB status
    def get_chroma_status(self):
        return {
            'regulations': self.chroma.get_document_count("regulations"),
            'policies': self.chroma.get_document_count("policies")
        }


#  existing helper functions...
def get_all_pdf_files():
    reg_files = list(Path("data/regulations").glob("*.pdf"))
    pol_files = list(Path("data/policies").glob("*.pdf"))
    return [str(f) for f in (reg_files + pol_files)]


#  UPDATED CLI with ChromaDB integration
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process PDFs to ChromaDB")
    parser.add_argument("--json", "-j", action="store_true", help="Also save to JSON")
    parser.add_argument("--output", "-o", default="data/processed/chunks.json")
    
    args = parser.parse_args()
    
    files = get_all_pdf_files()
    if not files:
        print(" No PDF files found in data/regulations or data/policies")
        exit(1)
    
    print(f" Found {len(files)} PDF files\n")
    
    pipeline = DocumentPipeline()
    results = pipeline.process_batch(files)
    
    # Summary
    stats = pipeline.get_stats()
    chroma_status = pipeline.get_chroma_status()
    
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"PDFs processed: {stats['processed']}")
    print(f"Failed: {stats['failed']}")
    print(f"Total pages: {stats['pages']}")
    print(f"Total chunks: {stats['chunks']}")
    print(f"\nChromaDB status:")
    print(f"  Regulations: {chroma_status['regulations']}")
    print(f"  Policies: {chroma_status['policies']}")
    print(f"  Total: {chroma_status['regulations'] + chroma_status['policies']}")
    
    # Optional JSON export
    if args.json:
        all_chunks = []
        for r in results:
            if r['success']:
                # Re-load from file or store during processing
                pass  # Implement if needed
        
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        # with open(args.output, 'w') as f:
        #     json.dump(all_chunks, f, indent=2)
        print(f"\n JSON export: {args.output} (not implemented)")
    
    print(f"\n Ready for gap analysis!")
    print(f"   Test: python -c \"from src.scoring.gap_analyzer import GapAnalyzer; ...\"")