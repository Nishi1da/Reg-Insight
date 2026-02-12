"""Document Chunker - Split into semantic chunks"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import re
import hashlib


class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
    
    def detect_section_header(self, text: str) -> str:
        """Detect section headers"""
        patterns = [
            r'^(?:Section|Article)\s+\d+[.:]?\s*(.+)$',
            r'^\d+[.:]\s+(.+)$',
            r'^[A-Z][.):]\s+(.+)$',
        ]
        
        for pattern in patterns:
            match = re.match(pattern, text.strip(), re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(0).strip()
        
        return ""
    
    def create_chunk_id(self, source: str, page: int, index: int) -> str:
        """Create unique chunk ID"""
        id_string = f"{source}_{page}_{index}"
        return f"chunk_{hashlib.md5(id_string.encode()).hexdigest()[:12]}"
    
    def chunk_page(self, page_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split a page into chunks"""
        text = page_data['content']
        source = page_data['document_metadata']['file_name']
        page_number = page_data['page_number']
        
        header = self.detect_section_header(text[:500])
        texts = self.text_splitter.split_text(text)
        
        chunks = []
        for idx, chunk_text in enumerate(texts):
            chunk = {
                'chunk_id': self.create_chunk_id(source, page_number, idx),
                'content': chunk_text,
                'source': source,
                'page_number': page_number,
                'chunk_index': idx,
                'total_chunks_on_page': len(texts),
                'section_header': header if idx == 0 else "",
                'metadata': {
                    'chunk_size': len(chunk_text),
                    'word_count': len(chunk_text.split())
                }
            }
            chunks.append(chunk)
        
        return chunks
    
    def chunk_document(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk all pages"""
        all_chunks = []
        
        for page in pages:
            page_chunks = self.chunk_page(page)
            all_chunks.extend(page_chunks)
        
        # Add global index
        for global_idx, chunk in enumerate(all_chunks):
            chunk['global_chunk_index'] = global_idx
            chunk['total_chunks'] = len(all_chunks)
        
        return all_chunks


# Test
if __name__ == "__main__":
    from pathlib import Path
    from pdf_loader import PDFLoader
    
    test_dir = Path("data/sample")
    pdf_files = list(test_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDFs found")
        exit(0)
    
    loader = PDFLoader()
    chunker = DocumentChunker()
    
    for pdf_file in pdf_files:
        print(f"\n{'='*60}")
        print(f"Processing: {pdf_file.name}")
        print('='*60)
        
        pages = loader.load(str(pdf_file))
        chunks = chunker.chunk_document(pages)
        
        print(f"Created {len(chunks)} chunks")
        
        if chunks:
            first = chunks[0]
            print(f"\nFirst chunk:")
            print(f"  ID: {first['chunk_id']}")
            print(f"  Page: {first['page_number']}")
            print(f"  Section: {first['section_header'] or '(none)'}")
            print(f"  Preview: {first['content'][:150]}...")