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

        self.stats = {
            'chunks_processed': 0,
            'chunks_failed': 0,
            'batches_processed': 0,
            'embedding_time': 0,
            'db_time': 0,
            'start_time': None
        }

    def ingest_chunks(
        self,
        chunks: List[Dict],
        collection_name: str = "regulations",
        show_progress: bool = True,
    ) -> Dict:

        self.stats['start_time'] = time.time()

        self.chroma_manager.create_collection(collection_name)
        logger.info(f"Starting ingestion of {len(chunks)} chunks into '{collection_name}'")

        num_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
        iterator = tqdm(range(num_batches), desc="Ingesting") if show_progress else range(num_batches)

        for batch_idx in iterator:
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(chunks))
            batch_chunks = chunks[start_idx:end_idx]

            try:
                texts = [c['content'] for c in batch_chunks]

                t0 = time.time()
                embeddings = self.embedding_gen.encode(
                    texts,
                    batch_size=self.batch_size,
                    show_progress=False
                )
                self.stats['embedding_time'] += time.time() - t0

                metadatas = []
                ids = []

                for i, chunk in enumerate(batch_chunks):

                    nested = chunk.get('metadata', {}) if isinstance(chunk.get('metadata'), dict) else {}

                    chunk_id = chunk.get('chunk_id') or chunk.get('id') or f"chunk_{start_idx+i}"
                    source = chunk.get('source') or nested.get('source') or 'unknown'
                    page = chunk.get('page_number') or nested.get('page_number') or 0
                    section = chunk.get('section_header') or nested.get('section_header') or ''

                    meta = {
                        'chunk_id': chunk_id,
                        'source': source,
                        'page_number': int(page),
                        'section_header': section,
                        'word_count': nested.get('word_count', 0),
                        'chunk_size': nested.get('chunk_size', 0),
                        'ingested_at': datetime.now().isoformat()
                    }

                    metadatas.append(meta)
                    ids.append(chunk_id)

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

                for chunk in batch_chunks:
                    try:
                        self._ingest_single(chunk, collection_name)
                        self.stats['chunks_processed'] += 1
                    except Exception as e2:
                        logger.error(f"Failed to ingest chunk: {e2}")
                        self.stats['chunks_failed'] += 1

        total_time = time.time() - self.stats['start_time']
        logger.info(f"Ingestion complete: {self.stats['chunks_processed']} chunks in {total_time:.1f}s")

        return self.get_stats()

    def _ingest_single(self, chunk: Dict, collection_name: str):

        chunk_meta = chunk.get('metadata', {}) if isinstance(chunk.get('metadata'), dict) else {}

        chunk_id = chunk.get('chunk_id') or chunk.get('id') or f"single_{int(time.time()*1000)}"

        meta = {
            'chunk_id': chunk_id,
            'source': chunk.get('source') or chunk_meta.get('source', 'unknown'),
            'page_number': chunk.get('page_number') or chunk_meta.get('page_number', 0),
            'section_header': chunk.get('section_header') or chunk_meta.get('section_header', ''),
            'word_count': chunk_meta.get('word_count', 0),
            'chunk_size': chunk_meta.get('chunk_size', 0),
            'ingested_at': datetime.now().isoformat()
        }

        embedding = self.embedding_gen.encode(chunk['content'])

        self.chroma_manager.add_documents(
            documents=[chunk['content']],
            embeddings=[embedding.tolist()],
            metadatas=[meta],
            ids=[chunk_id],
            collection_name=collection_name
        )

    def ingest_from_pipeline(
        self,
        pdf_paths: List[str],
        collection_name: str = "regulations",
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ) -> Dict:

        logger.info("Step 1: Processing PDFs...")
        doc_pipeline = DocumentPipeline(chunk_size, chunk_overlap)
        results = doc_pipeline.process_batch(pdf_paths)

        all_chunks = []
        for result in results:
            if result['success']:
                all_chunks.extend(result['chunks'])

        logger.info(f"Extracted {len(all_chunks)} chunks from {len(pdf_paths)} PDFs")

        return self.ingest_chunks(all_chunks, collection_name)

    def get_stats(self) -> Dict:
        stats = self.stats.copy()
        if stats['start_time']:
            stats['total_time'] = time.time() - stats['start_time']
            stats['chunks_per_second'] = stats['chunks_processed'] / stats['total_time']
        return stats