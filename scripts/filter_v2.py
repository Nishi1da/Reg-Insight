import sys
sys.path.insert(0, 'src')

from pathlib import Path
from ingestion.pdf_loader import PDFLoader
from ingestion.chunker import DocumentChunker
from embeddings.chroma_manager import ChromaManager
from embeddings.embedding_generator import EmbeddingGenerator

# Stricter filters
MIN_CHUNK_LENGTH = 200  # Increased from 150
MIN_SENTENCES = 2  # Must have at least 2 sentences
BAD_PATTERNS = [
    r'^[A-Z][A-Z\s]{2,}$',  # All caps lines like "AML CFT"
    r'^([A-Z][a-z]+)\s+([A-Z][a-z]+)\s*$',  # "Term Definition" pattern
]

TOC_KEYWORDS = ['contents', 'table of contents', 'purpose of the guidelines', 
                'acronyms', 'abbreviations', 'page ', '..........', '.................',
                'chapter i', 'preliminary', 'short title and commencement']

def is_quality_chunk(chunk_text, section_header):
    """Stricter quality check"""
    text = chunk_text.strip()
    
    # Too short
    if len(text) < MIN_CHUNK_LENGTH:
        return False, "too_short"
    
    # Count sentences (rough)
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
    if len(sentences) < MIN_SENTENCES:
        return False, "too_few_sentences"
    
    # Too many dots (TOC)
    if text.count('.') > len(text) * 0.25:
        return False, "toc_dots"
    
    # Too many newlines (lists/headers)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) > 5 and all(len(l) < 50 for l in lines):
        return False, "list_format"
    
    # Check for definition list pattern (Acronym\nFull Form)
    definition_count = 0
    for i, line in enumerate(lines[:-1]):
        if len(line) < 10 and len(lines[i+1]) > 20 and line.isupper():
            definition_count += 1
    if definition_count >= 3:  # 3+ definitions = acronym list
        return False, "acronym_list"
    
    # TOC keywords
    text_lower = text.lower()
    first_200 = text_lower[:200]
    for keyword in TOC_KEYWORDS:
        if keyword in first_200:
            return False, f"toc_keyword: {keyword}"
    
    # Just a header with no body
    if len(lines) <= 3 and section_header:
        return False, "header_only"
    
    return True, "quality"

import re

def filter_and_reindex():
    print("=" * 70)
    print("FILTER V2 - Stricter Quality Control")
    print("=" * 70)
    
    loader = PDFLoader()
    chunker = DocumentChunker(chunk_size=512, chunk_overlap=50)
    embedder = EmbeddingGenerator()
    cm = ChromaManager()
    
    # Delete and recreate
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
    print(f"  Removed reasons: {stats['regulations']['removed']}")
    
    print(f"\nPolicies: {stats['policies']['kept']}/{stats['policies']['total']} kept")
    print(f"  Removed reasons: {stats['policies']['removed']}")
    
    print(f"\n Final database: {reg_final} regulation chunks, {pol_final} policy chunks")
    
    # Show 3 samples
    if reg_final > 0:
        coll = cm.get_collection('regulations')
        samples = coll.get(limit=3)
        print(f"\nSample quality chunks:")
        for i, (doc, meta) in enumerate(zip(samples['documents'], samples['metadatas'])):
            print(f"\n  Sample {i+1}:")
            print(f"    ID: {meta['chunk_id']}")
            print(f"    Source: {meta['source']} p.{meta['page']}")
            print(f"    Content: {doc[:180]}...")

if __name__ == '__main__':
    filter_and_reindex()