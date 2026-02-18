"""Ingestion Pipeline - Chunks to Vector Store"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Optional
import numpy as np
from tqdm import tqdm
import json
import time
import logging
from datetime import datetime

from ingestion.document_pipeline import DocumentPipeline
from embeddings.embedding_generator import EmbeddingGenerator
from embeddings.chroma_manager import ChromaManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchIngestionPipeline:
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chroma_path: str = "data/processed/chroma_db",
        batch_size: int = 32,
        use_cache: bool = True
    ):
        self.embedding_gen = EmbeddingGenerator(
            model_name=embedding_model,
            use_cache=use_cache
        )
        self.chroma_manager = ChromaManager(
            persist_directory=chroma_path
        )
        self.batch_size = batch_size
        
        # Statistics tracking
        self.stats = {
            'chunks_processed': 0,
            'chunks_failed': 0,
            'batches_processed': 0,
            'embedding_time': 0,
            'db_time': 0,
            'start_time': None
        }
    
    def _optimize_batch_size(self, available_memory_gb: float = None) -> int:
        """Auto-optimize batch size based on available RAM"""
        if available_memory_gb is None:
            try:
                import psutil
                available_memory_gb = psutil.virtual_memory().available / (1024**3)
            except ImportError:
                return self.batch_size
        
        # Rough heuristic: 384 dim * 4 bytes * batch_size * safety_factor < available_memory/4
        estimated_batch = int((available_memory_gb * 0.25 * 1024**3) / (384 * 4 * 2))
        optimized = min(max(estimated_batch, 8), 256)  # Clamp between 8 and 256
        
        logger.info(f"Optimized batch size: {optimized} (available RAM: {available_memory_gb:.1f}GB)")
        return optimized
    
    def _compute_embedding_stats(self, embeddings: np.ndarray) -> Dict:
        """Compute statistics for monitoring"""
        return {
            'mean': float(np.mean(embeddings)),
            'std': float(np.std(embeddings)),
            'min': float(np.min(embeddings)),
            'max': float(np.max(embeddings)),
            'norm_mean': float(np.mean([np.linalg.norm(e) for e in embeddings]))
        }
    
    def ingest_chunks(
        self,
        chunks: List[Dict],
        collection_name: str = "regulations",
        show_progress: bool = True,
        metadata_fields: Optional[List[str]] = None
    ) -> Dict:
        """
        Ingest chunks into vector store
        
        Args:
            chunks: List of chunk dictionaries from chunker
            collection_name: Target ChromaDB collection
            show_progress: Show progress bars
            metadata_fields: Additional fields to include in metadata
        """
        self.stats['start_time'] = time.time()
        metadata_fields = metadata_fields or ['chunk_id', 'source', 'page_number', 'section_header']
        
        # Ensure collection exists
        self.chroma_manager.create_collection(collection_name)
        
        logger.info(f"Starting ingestion of {len(chunks)} chunks into '{collection_name}'")
        
        # Process in batches
        num_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
        
        iterator = tqdm(range(num_batches), desc="Ingesting") if show_progress else range(num_batches)
        
        for batch_idx in iterator:
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(chunks))
            batch_chunks = chunks[start_idx:end_idx]
            
            try:
                # Extract texts
                texts = [c['content'] for c in batch_chunks]
                
                # Generate embeddings
                t0 = time.time()
                embeddings = self.embedding_gen.encode(
                    texts,
                    batch_size=self.batch_size,
                    show_progress=False
                )
                self.stats['embedding_time'] += time.time() - t0
                
                # Prepare metadata
                metadatas = []
                for chunk in batch_chunks:
                    meta = {field: chunk.get(field, '') for field in metadata_fields}
                    meta.update({
                        'word_count': chunk.get('metadata', {}).get('word_count', 0),
                        'chunk_size': chunk.get('metadata', {}).get('chunk_size', 0),
                        'ingested_at': datetime.now().isoformat()
                    })
                    metadatas.append(meta)
                
                # Prepare IDs
                ids = [c['chunk_id'] for c in batch_chunks]
                
                # Add to ChromaDB
                t0 = time.time()
                self.chroma_manager.add_documents(
                    documents=texts,
                    embeddings=embeddings.tolist(),
                    metadatas=metadatas,
                    ids=ids,
                    collection_name=collection_name
                )
                self.stats['db_time'] += time.time() - t0
                
                self.stats['chunks_processed'] += len(batch_chunks)
                self.stats['batches_processed'] += 1
                
            except Exception as e:
                logger.error(f"Batch {batch_idx} failed: {e}")
                self.stats['chunks_failed'] += len(batch_chunks)
                
                # Fallback: try one by one
                for chunk in batch_chunks:
                    try:
                        self._ingest_single(chunk, collection_name)
                        self.stats['chunks_processed'] += 1
                    except Exception as e2:
                        logger.error(f"Failed to ingest chunk {chunk.get('chunk_id')}: {e2}")
                        self.stats['chunks_failed'] += 1
        
        total_time = time.time() - self.stats['start_time']
        logger.info(f"Ingestion complete: {self.stats['chunks_processed']} chunks in {total_time:.1f}s")
        
        return self.get_stats()
    
    def _ingest_single(self, chunk: Dict, collection_name: str):
        """Fallback: ingest single chunk"""
        embedding = self.embedding_gen.encode(chunk['content'])
        self.chroma_manager.add_documents(
            documents=[chunk['content']],
            embeddings=[embedding.tolist()],
            metadatas=[{
                'chunk_id': chunk['chunk_id'],
                'source': chunk['source'],
                'page_number': chunk['page_number']
            }],
            ids=[chunk['chunk_id']],
            collection_name=collection_name
        )
    
    def ingest_from_pipeline(
        self,
        pdf_paths: List[str],
        collection_name: str = "regulations",
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ) -> Dict:
        """
        Full pipeline: PDFs → Chunks → Embeddings → Vector Store
        """
        # Step 1: Process PDFs
        logger.info("Step 1: Processing PDFs...")
        doc_pipeline = DocumentPipeline(chunk_size, chunk_overlap)
        results = doc_pipeline.process_batch(pdf_paths)
        
        # Collect all chunks
        all_chunks = []
        for result in results:
            if result['success']:
                all_chunks.extend(result['chunks'])
        
        logger.info(f"Extracted {len(all_chunks)} chunks from {len(pdf_paths)} PDFs")
        
        # Step 2: Ingest to vector store
        return self.ingest_chunks(all_chunks, collection_name)
    
    def get_stats(self) -> Dict:
        """Get ingestion statistics"""
        stats = self.stats.copy()
        if stats['start_time']:
            stats['total_time'] = time.time() - stats['start_time']
            stats['chunks_per_second'] = stats['chunks_processed'] / stats['total_time']
        return stats
    
    def export_stats(self, filepath: str = "outputs/ingestion_stats.json"):
        """Export statistics to JSON"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.get_stats(), f, indent=2)


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Day 10: Batch Ingestion Pipeline Test")
    print("=" * 60)
    
    # Check if we have sample data
    sample_dir = Path("data/sample")
    if not sample_dir.exists() or not list(sample_dir.glob("*.pdf")):
        print(" No sample PDFs found. Creating test with dummy data...")
        
        # Test with dummy chunks
        pipeline = BatchIngestionPipeline()
        
        dummy_chunks = [
            {
                'chunk_id': f'test_chunk_{i}',
                'content': f'This is test regulation content about topic {i}. ' * 20,
                'source': 'test.pdf',
                'page_number': i % 5 + 1,
                'section_header': f'Section {i}',
                'metadata': {'word_count': 100, 'chunk_size': 500}
            }
            for i in range(10)
        ]
        
        stats = pipeline.ingest_chunks(dummy_chunks, "test_collection")
        print(f"\nStats: {json.dumps(stats, indent=2)}")
        
    else:
        # Full test with real PDFs
        print("\n1. Testing full pipeline...")
        pipeline = BatchIngestionPipeline(batch_size=4)
        
        pdf_files = list(sample_dir.glob("*.pdf"))
        stats = pipeline.ingest_from_pipeline(
            [str(f) for f in pdf_files],
            "regulations"
        )
        
        print(f"\nFinal stats:")
        print(f"  Chunks processed: {stats['chunks_processed']}")
        print(f"  Failed: {stats['chunks_failed']}")
        print(f"  Total time: {stats.get('total_time', 0):.1f}s")
        print(f"  Speed: {stats.get('chunks_per_second', 0):.1f} chunks/sec")
        
        # Verify
        count = pipeline.chroma_manager.get_document_count("regulations")
        print(f"\n2. Verification: {count} documents in collection")
    
    print("\n" + "=" * 60)
    print(" Day 10 complete!")
    print("=" * 60)