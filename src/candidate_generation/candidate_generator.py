"""Candidate Generator - Regulation to Policy Matching"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import logging
from datetime import datetime

from embeddings.chroma_manager import ChromaManager
from embeddings.embedding_generator import EmbeddingGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CandidatePair:
    """Represents a regulation-policy candidate match"""
    regulation_chunk_id: str
    regulation_text: str
    regulation_metadata: Dict

    policy_chunk_id: str
    policy_text: str
    policy_metadata: Dict

    bi_encoder_score: float
    rank: int

    def to_dict(self) -> Dict:
        return {
            'regulation_chunk_id': self.regulation_chunk_id,
            'regulation_text': self.regulation_text[:200],
            'policy_chunk_id': self.policy_chunk_id,
            'policy_text': self.policy_text[:200],
            'bi_encoder_score': self.bi_encoder_score,
            'rank': self.rank,
            'regulation_source': self.regulation_metadata.get('source'),
            'policy_source': self.policy_metadata.get('source'),
            'regulation_page': self.regulation_metadata.get('page_number'),
            'policy_page': self.policy_metadata.get('page_number')
        }


class CandidateGenerator:
    def __init__(
        self,
        chroma_path: str = "data/processed/chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        reg_collection: str = "regulations",
        policy_collection: str = "policies"
    ):
        self.chroma = ChromaManager(persist_directory=chroma_path)
        self.embedder = EmbeddingGenerator(model_name=embedding_model)
        self.reg_collection = reg_collection
        self.policy_collection = policy_collection

        self.stats = {
            'chunks_processed': 0,
            'candidates_generated': 0,
            'avg_candidates_per_reg': 0.0,
            'processing_time': 0.0
        }

    def get_candidates(
        self,
        regulation_chunk: Dict,
        top_k: int = 5,
        min_score: float = 0.05,
        where_filter: Optional[Dict] = None
    ) -> List[CandidatePair]:
        """
        Find top-k policy candidates for a regulation chunk.

        Args:
            regulation_chunk : Dict with 'content', 'metadata', 'chunk_id'
            top_k            : Number of candidates to return
            min_score        : Minimum similarity threshold
            where_filter     : Optional ChromaDB metadata filter dict.
                               If None, queries the full policy collection.
                               Example (domain-filtered):
                                 {"$and": [
                                     {"doc_type": {"$eq": "policy"}},
                                     {"domain":   {"$eq": "aml"}}
                                 ]}

        Returns:
            List of CandidatePair objects sorted by similarity (descending)
        """
        # Embed regulation chunk
        reg_embedding = self.embedder.encode(regulation_chunk['content'])

        print(f"\n[DEBUG] Querying for regulation: {regulation_chunk['chunk_id'][:30]}...")
        print(f"[DEBUG] Embedding dim: {len(reg_embedding)}")
        print(f"[DEBUG] Searching collection: '{self.policy_collection}'")
        if where_filter:
            print(f"[DEBUG] Where filter: {where_filter}")

        # Build query kwargs — only add where if a filter was supplied
        query_kwargs = {
            "query_embeddings": [reg_embedding.tolist()],
            "n_results":        top_k * 2,   # fetch more, filter down after
            "collection_name":  self.policy_collection,
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        # Execute query
        try:
            results = self.chroma.query(**query_kwargs)
        except Exception as e:
            # If the filtered query fails (e.g. no matching docs), fall back
            logger.debug(f"Filtered query failed ({e}), retrying without filter")
            query_kwargs.pop("where", None)
            results = self.chroma.query(**query_kwargs)

        print(f"[DEBUG] Raw results: {len(results['ids'][0]) if results['ids'] else 0} candidates")
        if results['ids'] and results['ids'][0]:
            print(f"[DEBUG] Top distance: {results['distances'][0][0]:.4f}")

        candidates = []

        if not results['ids'] or not results['ids'][0]:
            return candidates

        for i, (policy_id, policy_text, policy_meta, distance) in enumerate(zip(
            results['ids'][0],
            results['documents'][0],
            results['metadatas'][0],
            results.get('distances', [[]])[0]
        )):
            # Skip self-match
            if policy_id == regulation_chunk.get('chunk_id'):
                continue

            if distance is None:
                similarity = 0.0
            else:
                similarity = 1.0 / (1.0 + distance)

            if similarity < min_score:
                continue

            candidate = CandidatePair(
                regulation_chunk_id=regulation_chunk['chunk_id'],
                regulation_text=regulation_chunk['content'],
                regulation_metadata=regulation_chunk['metadata'],
                policy_chunk_id=policy_id,
                policy_text=policy_text,
                policy_metadata=policy_meta,
                bi_encoder_score=float(similarity),
                rank=len(candidates) + 1
            )
            candidates.append(candidate)

            if len(candidates) >= top_k:
                break

        return candidates

    def generate_for_collection(
        self,
        collection_name: str = "regulations",
        top_k: int = 3,
        min_score: float = 0.3
    ) -> List[CandidatePair]:
        """
        Generate candidates for all regulation chunks in a collection.

        Args:
            collection_name : Source collection (regulations)
            top_k           : Candidates per regulation chunk
            min_score       : Minimum similarity threshold

        Returns:
            List of all CandidatePairs
        """
        start_time = datetime.now()

        collection = self.chroma.get_collection(collection_name)
        all_data   = collection.get()

        logger.info(f"Processing {len(all_data['ids'])} regulation chunks...")

        all_candidates = []

        for i, (chunk_id, text, metadata) in enumerate(zip(
            all_data['ids'], all_data['documents'], all_data['metadatas']
        )):
            if i % 50 == 0:
                logger.info(f"  Processed {i}/{len(all_data['ids'])} chunks...")

            reg_chunk = {
                'chunk_id': chunk_id,
                'content':  text,
                'metadata': metadata
            }

            candidates = self.get_candidates(reg_chunk, top_k, min_score)
            all_candidates.extend(candidates)

        elapsed = (datetime.now() - start_time).total_seconds()
        self.stats = {
            'chunks_processed':      len(all_data['ids']),
            'candidates_generated':  len(all_candidates),
            'avg_candidates_per_reg': (
                len(all_candidates) / len(all_data['ids'])
                if all_data['ids'] else 0
            ),
            'processing_time': elapsed
        }

        logger.info(f"Generated {len(all_candidates)} candidates in {elapsed:.1f}s")
        logger.info(
            f"Average candidates per regulation: "
            f"{self.stats['avg_candidates_per_reg']:.2f}"
        )

        return all_candidates

    def get_stats(self) -> Dict:
        return self.stats.copy()

    def save_candidates(self, candidates: List[CandidatePair], filepath: str):
        import json
        data = {
            'generated_at': datetime.now().isoformat(),
            'stats':        self.stats,
            'candidates':   [c.to_dict() for c in candidates]
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(candidates)} candidates to {filepath}")


# ------------------------------------------------------------------ #
#  Quick smoke-test                                                    #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("=" * 60)
    print("Candidate Generator — smoke test")
    print("=" * 60)

    gen   = CandidateGenerator()
    chroma = ChromaManager()
    coll  = chroma.get_collection("regulations")
    sample = coll.get(limit=1)

    if sample['ids']:
        reg_chunk = {
            'chunk_id': sample['ids'][0],
            'content':  sample['documents'][0],
            'metadata': sample['metadatas'][0]
        }

        print(f"\nRegulation: {reg_chunk['content'][:80]}...")

        # Test unfiltered
        candidates = gen.get_candidates(reg_chunk, top_k=5, min_score=0.15)
        print(f"\nUnfiltered — {len(candidates)} candidates:")
        for c in candidates:
            print(f"  Rank {c.rank}: score={c.bi_encoder_score:.3f} | "
                  f"{c.policy_text[:60]}...")

        # Test domain-filtered
        OP_EQ  = "$eq"
        OP_AND = "$and"
        domain = reg_chunk['metadata'].get('domain', 'general')
        filtered = gen.get_candidates(
            reg_chunk,
            top_k=5,
            min_score=0.15,
            where_filter={
                OP_AND: [
                    {"doc_type": {OP_EQ: "policy"}},
                    {"domain":   {OP_EQ: domain}}
                ]
            }
        )
        print(f"\nDomain-filtered (domain='{domain}') — {len(filtered)} candidates:")
        for c in filtered:
            print(f"  Rank {c.rank}: score={c.bi_encoder_score:.3f} | "
                  f"{c.policy_text[:60]}...")

    print("\n" + "=" * 60)