import sys
sys.path.insert(0, 'src')

from pathlib import Path
from ingestion.pdf_loader import PDFLoader
from ingestion.chunker import DocumentChunker  # ADD THIS LINE
from embeddings.chroma_manager import ChromaManager
from embeddings.embedding_generator import EmbeddingGenerator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_database():
    cm = ChromaManager()
    loader = PDFLoader()
    chunker = DocumentChunker(chunk_size=512, chunk_overlap=50)  # ADD THIS LINE
    embedder = EmbeddingGenerator()
    
    # Create collections first
    print('=== CREATING COLLECTIONS ===')
    cm.create_collection('regulations', metadata={'type': 'regulations'})
    cm.create_collection('policies', metadata={'type': 'policies'})
    print('Collections created')
    
    print('\n=== INDEXING REGULATIONS ===')
    reg_dir = Path('data/regulations')
    for pdf in reg_dir.glob('*.pdf'):
        pages = loader.load(str(pdf))              # CHANGE: pages (not chunks)
        chunks = chunker.chunk_document(pages)      # ADD THIS LINE
        print(f'{pdf.name}: {len(chunks)} chunks')  # Now shows ~120, not 30!
        if chunks:
            texts = [c['content'] for c in chunks]
            embeddings = embedder.encode(texts, batch_size=32, show_progress=False)
            
            cm.add_documents(
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=[{
                    'source': pdf.name, 
                    'page': c['page_number'],
                    'chunk_id': c['chunk_id']  # Optional: better tracking
                } for c in chunks],
                ids=[c['chunk_id'] for c in chunks],  # Use proper chunk IDs
                collection_name='regulations'
            )

    print('\n=== INDEXING POLICIES ===')
    pol_dir = Path('data/policies')
    for pdf in pol_dir.glob('*.pdf'):
        pages = loader.load(str(pdf))              # CHANGE: pages (not chunks)
        chunks = chunker.chunk_document(pages)      # ADD THIS LINE
        print(f'{pdf.name}: {len(chunks)} chunks')  # Now shows ~140, not 35!
        if chunks:
            texts = [c['content'] for c in chunks]
            embeddings = embedder.encode(texts, batch_size=32, show_progress=False)
            
            cm.add_documents(
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=[{
                    'source': pdf.name, 
                    'page': c['page_number'],
                    'chunk_id': c['chunk_id']
                } for c in chunks],
                ids=[c['chunk_id'] for c in chunks],  # Use proper chunk IDs
                collection_name='policies'
            )

    print('\n=== VERIFICATION ===')
    reg_count = cm.get_collection('regulations').count()
    pol_count = cm.get_collection('policies').count()
    print(f'Regulations: {reg_count} chunks')
    print(f'Policies: {pol_count} chunks')
    
    total = reg_count + pol_count
    print(f'Total: {total} chunks (expected: ~500-600)')
    
    if total > 400:  # Should be ~530
        print('\n✅ SUCCESS! Database rebuilt with semantic chunking.')
        print('\nNext: Test your GapAnalyzer')
        print('  python src/scoring/gap_analyzer.py -a -l 20 -s')
    else:
        print(f'\n❌ ERROR: Only {total} chunks - chunking may have failed!')

if __name__ == '__main__':
    fix_database()