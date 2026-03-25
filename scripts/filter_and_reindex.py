import sys
sys.path.insert(0, 'src')

from pathlib import Path
from ingestion.pdf_loader import PDFLoader
from ingestion.chunker import DocumentChunker
from embeddings.chroma_manager import ChromaManager
from embeddings.embedding_generator import EmbeddingGenerator
import re

# Quality filters
MIN_CHUNK_LENGTH = 150  # Minimum characters
MAX_HEADER_RATIO = 0.5  # Max ratio of header-like lines
TOC_KEYWORDS = ['contents', 'table of contents', 'purpose of the guidelines', 
                'acronyms', 'page ', '..........', '.................']

def is_quality_chunk(chunk_text, section_header):
    """Check if chunk is substantive (not TOC/header/title page)"""
    text = chunk_text.strip()
    
    # Too short
    if len(text) < MIN_CHUNK_LENGTH:
        return False, "too_short"
    
    # Mostly dots (TOC entries)
    if text.count('.') > len(text) * 0.3:
        return False, "toc_dots"
    
    # TOC keywords
    text_lower = text.lower()
    for keyword in TOC_KEYWORDS:
        if keyword in text_lower:
            return False, f"toc_keyword: {keyword}"
    
    # Just a header with no body
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) <= 2 and section_header:
        return False, "header_only"
    
    # Title page pattern (short lines, centered-ish)
    if len(lines) < 5 and all(len(l) < 50 for l in lines):
        return False, "title_page"
    
    return True, "quality"

def filter_and_reindex():
    print("=" * 70)
    print("FILTER & REINDEX - Removing Low-Quality Chunks")
    print("=" * 70)
    
    loader = PDFLoader()
    chunker = DocumentChunker(chunk_size=512, chunk_overlap=50)
    embedder = EmbeddingGenerator()
    cm = ChromaManager()
    
    # Delete old collections
    print("\n=== CLEANING ===")
    try:
        cm.delete_collection('regulations')
        cm.delete_collection('policies')
        print(" Deleted old collections")
    except:
        pass
    
    cm.create_collection('regulations', metadata={'type': 'regulations'})
    cm.create_collection('policies', metadata={'type': 'policies'})
    
    stats = {'regulations': {'total': 0, 'kept': 0, 'removed': {}},
             'policies': {'total': 0, 'kept': 0, 'removed': {}}}
    
    # Process regulations
    print("\n=== FILTERING REGULATIONS ===")
    for pdf in Path('data/regulations').glob('*.pdf'):
        print(f"\n {pdf.name}")
        pages = loader.load(str(pdf))
        chunks = chunker.chunk_document(pages)
        
        quality_chunks = []
        for chunk in chunks:
            is_good, reason = is_quality_chunk(chunk['content'], chunk.get('section_header', ''))
            stats['regulations']['total'] += 1
            
            if is_good:
                quality_chunks.append(chunk)
                stats['regulations']['kept'] += 1
            else:
                stats['regulations']['removed'][reason] = stats['regulations']['removed'].get(reason, 0) + 1
        
        print(f"   Total: {len(chunks)} | Kept: {len(quality_chunks)} | Removed: {len(chunks) - len(quality_chunks)}")
        
        if quality_chunks:
            texts = [c['content'] for c in quality_chunks]
            embeddings = embedder.encode(texts, batch_size=32, show_progress=False)
            
            cm.add_documents(
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=[{
                    'source': c['source'],
                    'page': c['page_number'],
                    'section_header': c.get('section_header', ''),
                    'chunk_id': c['chunk_id']
                } for c in quality_chunks],
                ids=[c['chunk_id'] for c in quality_chunks],
                collection_name='regulations'
            )
    
    # Process policies
    print("\n=== FILTERING POLICIES ===")
    for pdf in Path('data/policies').glob('*.pdf'):
        print(f"\n {pdf.name}")
        pages = loader.load(str(pdf))
        chunks = chunker.chunk_document(pages)
        
        quality_chunks = []
        for chunk in chunks:
            is_good, reason = is_quality_chunk(chunk['content'], chunk.get('section_header', ''))
            stats['policies']['total'] += 1
            
            if is_good:
                quality_chunks.append(chunk)
                stats['policies']['kept'] += 1
            else:
                stats['policies']['removed'][reason] = stats['policies']['removed'].get(reason, 0) + 1
        
        print(f"   Total: {len(chunks)} | Kept: {len(quality_chunks)} | Removed: {len(chunks) - len(quality_chunks)}")
        
        if quality_chunks:
            texts = [c['content'] for c in quality_chunks]
            embeddings = embedder.encode(texts, batch_size=32, show_progress=False)
            
            cm.add_documents(
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=[{
                    'source': c['source'],
                    'page': c['page_number'],
                    'section_header': c.get('section_header', ''),
                    'chunk_id': c['chunk_id']
                } for c in quality_chunks],
                ids=[c['chunk_id'] for c in quality_chunks],
                collection_name='policies'
            )
    
    # Final stats
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    reg_final = cm.get_collection('regulations').count()
    pol_final = cm.get_collection('policies').count()
    
    print(f"\nRegulations: {stats['regulations']['kept']}/{stats['regulations']['total']} kept")
    print(f"  Removed: {stats['regulations']['removed']}")
    
    print(f"\nPolicies: {stats['policies']['kept']}/{stats['policies']['total']} kept")
    print(f"  Removed: {stats['policies']['removed']}")
    
    print(f"\n Final database: {reg_final} regulation chunks, {pol_final} policy chunks")
    print(f"   (Was 530 total, now {reg_final + pol_final} quality chunks)")
    
    # Show sample
    if reg_final > 0:
        coll = cm.get_collection('regulations')
        sample = coll.get(limit=1)
        print(f"\nSample quality chunk:")
        print(f"  ID: {sample['ids'][0]}")
        print(f"  Content: {sample['documents'][0][:200]}...")

if __name__ == '__main__':
    filter_and_reindex()