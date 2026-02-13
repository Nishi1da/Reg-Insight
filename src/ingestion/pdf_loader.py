"""PDF Loader - Extract text and metadata"""

import fitz
from pathlib import Path
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFLoader:
    def __init__(self):
        self.supported_extensions = {'.pdf'}
    
    def load(self, filepath: str) -> List[Dict[str, Any]]:
        """Load PDF and extract text with metadata"""
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {filepath}")
        
        if path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"File must be PDF, got: {path.suffix}")
        
        logger.info(f"Loading: {path.name}")
        
        documents = []
        
        try:
            with fitz.open(filepath) as doc:
                # Handle encrypted PDFs
                if doc.is_encrypted:
                    if not doc.authenticate(""):
                        raise ValueError("PDF is password protected")
                
                # Document metadata
                doc_metadata = {
                    'title': doc.metadata.get('title', ''),
                    'author': doc.metadata.get('author', ''),
                    'total_pages': len(doc),
                    'file_name': path.name,
                }
                
                # Extract each page
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text()
                    
                    # Skip empty pages
                    if not text.strip():
                        logger.warning(f"Page {page_num + 1} is empty")
                        continue
                    
                    page_data = {
                        'content': text,
                        'page_number': page_num + 1,
                        'total_pages': len(doc),
                        'width': page.rect.width,
                        'height': page.rect.height,
                        'document_metadata': doc_metadata
                    }
                    
                    documents.append(page_data)
                
                logger.info(f"Extracted {len(documents)} pages")
                
        except fitz.FileDataError as e:
            raise ValueError(f"Corrupted PDF: {e}")
        
        return documents


# Test
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    test_dir = Path("data/sample")
    pdf_files = list(test_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDFs in data/sample/")
        sys.exit(0)
    
    loader = PDFLoader()
    
    for pdf_file in pdf_files:
        print(f"\n{'='*60}")
        print(f"Testing: {pdf_file.name}")
        print('='*60)
        
        try:
            pages = loader.load(str(pdf_file))
            print(f" Loaded {len(pages)} pages")
            
            if pages:
                print(f"\nPage 1 preview:")
                print(pages[0]['content'][:200].replace('\n', ' '))
                print(f"\nMetadata:")
                print(f"  Size: {pages[0]['width']:.0f} x {pages[0]['height']:.0f}")
                
        except Exception as e:
            print(f" Error: {e}")