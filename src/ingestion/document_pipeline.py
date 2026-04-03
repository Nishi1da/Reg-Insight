"""Document Pipeline - End-to-end: PDF → Chunks → ChromaDB"""

from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from .pdf_loader import PDFLoader
from .chunker import DocumentChunker
from ..embeddings.chroma_manager import ChromaManager


class DocumentPipeline:
    def __init__(self, chunk_size=1200, chunk_overlap=0):
        self.loader = PDFLoader()
        self.chroma = ChromaManager()
        self.stats = {
            'processed': 0, 'failed': 0,
            'pages': 0, 'chunks': 0,
            'added_to_chroma': 0
        }

    def process_file(self, filepath: str) -> Dict[str, Any]:
        """Process single PDF and add to ChromaDB"""
        path = Path(filepath)

        # Determine doc_type from folder name
        if "regulations" in str(path).lower() or "regulation" in path.name.lower():
            doc_type = "regulation"
            collection_name = "regulations"
        elif "policies" in str(path).lower() or "policy" in path.name.lower():
            doc_type = "policy"
            collection_name = "policies"
        else:
            doc_type = "unknown"
            collection_name = "regulations"

        try:
            pages = self.loader.load(filepath)

            # Pass doc_type to chunker so it's embedded in every chunk
            chunker = DocumentChunker(document_type=doc_type)
            chunks = chunker.chunk_document(pages)

            if not chunks:
                self.stats['processed'] += 1
                return {
                    'success': True,
                    'source': path.name,
                    'pages': len(pages),
                    'chunks': 0,
                    'error': None
                }

            chroma_chunks = self._convert_to_chroma_format(chunks, path.name, doc_type)
            self._add_to_chroma(chroma_chunks, collection_name)

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
                'pages': 0,
                'chunks': 0,
                'error': str(e)
            }

    def _convert_to_chroma_format(
        self,
        chunks: List[Dict],
        filename: str,
        doc_type: str
    ) -> List[Dict]:
        """
        Convert chunk dicts to ChromaDB format.

        Crucially saves doc_type, source, page_number, section_header
        into metadata so gap_analyzer can filter by doc_type="regulation".
        """
        converted = []
        for i, chunk in enumerate(chunks):
            text = chunk.get('content') or chunk.get('text', '')
            if not text.strip():
                continue

            # Build metadata — every field that downstream code needs
            metadata = {
                'doc_type':       doc_type,                              # ← CRITICAL: enables filtering
                'source':         filename,
                'page_number':    chunk.get('page_number', 0),
                'chunk_index':    chunk.get('chunk_index', i),
                'section_header': chunk.get('section_header', ''),
                'word_count':     chunk.get('metadata', {}).get('word_count', len(text.split())),
                'domain': chunk.get('domain', 'general'),
                'domain_confidence': chunk.get('domain_confidence', 0.0),
            }

            converted.append({
                'text':     text,
                'metadata': metadata,
            })

        return converted

    def _add_to_chroma(self, chunks: List[Dict], collection_name: str):
        """Add converted chunks to ChromaDB collection."""
        if not chunks:
            return

        import uuid
        texts     = [c['text']     for c in chunks]
        metadatas = [c['metadata'] for c in chunks]
        ids       = [str(uuid.uuid4()) for _ in chunks]

        self.chroma.add_documents(
            documents=texts,
            embeddings=None,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name
        )

    def process_batch(self, filepaths: List[str], show_progress: bool = True):
        """Process multiple PDFs."""
        results = []
        iterator = tqdm(filepaths) if show_progress else filepaths
        for filepath in iterator:
            result = self.process_file(filepath)
            results.append(result)
        return results

    def get_stats(self):
        return self.stats.copy()

    def get_chroma_status(self):
        return {
            'regulations': self.chroma.get_document_count("regulations"),
            'policies':    self.chroma.get_document_count("policies"),
        }


def get_all_pdf_files():
    reg_files = list(Path("data/regulations").glob("*.pdf"))
    pol_files = list(Path("data/policies").glob("*.pdf"))
    return [str(f) for f in (reg_files + pol_files)]


if __name__ == "__main__":
    files = get_all_pdf_files()
    if not files:
        print("No PDF files found in data/regulations or data/policies")
        exit(1)

    print(f" Found {len(files)} PDF files\n")

    pipeline = DocumentPipeline()
    results = pipeline.process_batch(files)

    stats        = pipeline.get_stats()
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
    print(f"  Policies:    {chroma_status['policies']}")
    print(f"  Total:       {chroma_status['regulations'] + chroma_status['policies']}")
    print(f"\n Ready for gap analysis!")
    print(f"   Run: python -m src.scoring.gap_analyzer --analyze --summary --output outputs/thesis_gap_analysis4.json")