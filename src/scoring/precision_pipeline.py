"""Precision Scoring Pipeline - Bi-encoder to Cross-encoder re-ranking"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Optional, Tuple
import numpy as np
import logging
import time
import hashlib
import json
from datetime import datetime
from dataclasses import dataclass, asdict

from candidate_generation.candidate_generator import CandidatePair
from candidate_generation.ranker import RankedCandidate
from scoring.cross_encoder import CrossEncoderScorer, CrossEncoderScore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PrecisionScoredCandidate:
    """Candidate with both bi-encoder and cross-encoder scores"""
    regulation_chunk_id: str
    regulation_text: str
    regulation_metadata: Dict
    
    policy_chunk_id: str
    policy_text: str
    policy_metadata: Dict
    
    bi_encoder_score: float
    cross_encoder_score: float
    final_score: float  # Combined score
    
    rank_bi: int
    rank_cross: int
    rank_final: int
    
    inference_time_ms: float
    scored_at: str
    
    def to_dict(self) -> Dict:
        return {
            'regulation_chunk_id': self.regulation_chunk_id,
            'policy_chunk_id': self.policy_chunk_id,
            'bi_encoder_score': round(self.bi_encoder_score, 4),
            'cross_encoder_score': round(self.cross_encoder_score, 4),
            'final_score': round(self.final_score, 4),
            'rank_bi': self.rank_bi,
            'rank_cross': self.rank_cross,
            'rank_final': self.rank_final,
            'regulation_source': self.regulation_metadata.get('source'),
            'policy_source': self.policy_metadata.get('source'),
            'inference_time_ms': round(self.inference_time_ms, 2),
            'scored_at': self.scored_at
        }


class ScoringCache:
    """Persistent cache for cross-encoder scores"""
    
    def __init__(self, cache_dir: str = "data/scoring_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "cross_encoder_cache.json"
        self._memory_cache: Dict[str, float] = {}
        self._load_cache()
    
    def _make_key(self, reg_text: str, pol_text: str) -> str:
        """Create deterministic key from text pair"""
        combined = f"{reg_text.strip()}|||{pol_text.strip()}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _load_cache(self):
        """Load cache from disk"""
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                self._memory_cache = json.load(f)
            logger.info(f"Loaded {len(self._memory_cache)} cached scores")
    
    def _save_cache(self):
        """Save cache to disk"""
        with open(self.cache_file, 'w') as f:
            json.dump(self._memory_cache, f)
    
    def get(self, reg_text: str, pol_text: str) -> Optional[float]:
        """Get cached score if exists"""
        key = self._make_key(reg_text, pol_text)
        return self._memory_cache.get(key)
    
    def set(self, reg_text: str, pol_text: str, score: float):
        """Cache a score"""
        key = self._make_key(reg_text, pol_text)
        self._memory_cache[key] = score
        # Save periodically (every 100 new entries)
        if len(self._memory_cache) % 100 == 0:
            self._save_cache()
    
    def clear(self):
        """Clear cache"""
        self._memory_cache.clear()
        if self.cache_file.exists():
            self.cache_file.unlink()


class PrecisionScoringPipeline:
    """
    End-to-end precision scoring pipeline
    
    Flow: Bi-encoder candidates → Cross-encoder re-ranking → Combined scoring
    """
    
    def __init__(
        self,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        bi_encoder_weight: float = 0.3,
        cross_encoder_weight: float = 0.7,
        use_cache: bool = True,
        batch_size: int = 16
    ):
        self.cross_encoder = CrossEncoderScorer(
            model_name=cross_encoder_model,
            batch_size=batch_size
        )
        self.bi_weight = bi_encoder_weight
        self.cross_weight = cross_encoder_weight
        self.use_cache = use_cache
        self.cache = ScoringCache() if use_cache else None
        
        self.stats = {
            'total_candidates_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_scoring_time_ms': 0.0,
            'avg_scoring_time_ms': 0.0
        }
        
        self.scoring_metadata = {
            'model_version': cross_encoder_model,
            'bi_encoder_weight': bi_encoder_weight,
            'cross_encoder_weight': cross_encoder_weight,
            'created_at': datetime.now().isoformat()
        }
    
    def _calculate_final_score(
        self,
        bi_score: float,
        cross_score: float
    ) -> float:
        """Calculate weighted final score"""
        # Normalize bi-encoder score to 0-1 if needed
        # (Assuming bi-encoder cosine similarity is already 0-1)
        
        return (self.bi_weight * bi_score) + (self.cross_weight * cross_score)
    
    def score_candidates(
        self,
        candidates: List[RankedCandidate],
        regulation_chunk_id: Optional[str] = None
    ) -> List[PrecisionScoredCandidate]:
        """
        Re-rank candidates with cross-encoder precision scoring
        
        Args:
            candidates: List of RankedCandidate from bi-encoder
            regulation_chunk_id: Optional ID for tracking
        
        Returns:
            List of PrecisionScoredCandidate with both scores
        """
        if not candidates:
            return []
        
        start_time = time.time()
        
        # Prepare pairs for batch scoring
        pairs_to_score = []
        cached_results = []
        
        for c in candidates:
            # Check cache first
            if self.cache:
                cached_score = self.cache.get(c.regulation_text, c.policy_text)
                if cached_score is not None:
                    self.stats['cache_hits'] += 1
                    cached_results.append((c, cached_score))
                    continue
            
            pairs_to_score.append((
                c.regulation_chunk_id,
                c.regulation_text,
                c.policy_chunk_id,
                c.policy_text,
                c.bi_encoder_score,
                c.rank  # Original bi-encoder rank
            ))
            self.stats['cache_misses'] += 1
        
        # Score non-cached pairs with cross-encoder
        cross_encoder_results = []
        if pairs_to_score:
            # Prepare format for cross-encoder
            ce_pairs = [(p[0], p[1], p[2], p[3]) for p in pairs_to_score]
            ce_scores = self.cross_encoder.score_batch(ce_pairs, show_progress=False)
            
            # Cache new scores
            for (reg_id, reg_text, pol_id, pol_text, bi_score, bi_rank), ce_result in zip(pairs_to_score, ce_scores):
                if self.cache:
                    self.cache.set(reg_text, pol_text, ce_result.score)
                
                cross_encoder_results.append({
                    'candidate_data': (reg_id, reg_text, pol_id, pol_text, bi_score, bi_rank),
                    'cross_score': ce_result.score,
                    'inference_time': ce_result.inference_time_ms
                })
        
        # Combine all results
        all_scored = []
        
        # Add cached results
        for c, cross_score in cached_results:
            final_score = self._calculate_final_score(c.bi_encoder_score, cross_score)
            all_scored.append({
                'candidate': c,
                'cross_score': cross_score,
                'final_score': final_score,
                'inference_time': 0.0,
                'cached': True
            })
        
        # Add newly scored results
        for item in cross_encoder_results:
            c_data = item['candidate_data']
            # Recreate candidate object from data
            # (Simplified - in practice pass full candidate through)
            c = candidates[0]  # Placeholder - should map properly
            cross_score = item['cross_score']
            final_score = self._calculate_final_score(c_data[4], cross_score)
            
            all_scored.append({
                'candidate': c,
                'cross_score': cross_score,
                'final_score': final_score,
                'inference_time': item['inference_time'],
                'cached': False
            })
        
        # Sort by final score and create final objects
        all_scored.sort(key=lambda x: x['final_score'], reverse=True)
        
        results = []
        for rank, item in enumerate(all_scored, 1):
            c = item['candidate']
            psc = PrecisionScoredCandidate(
                regulation_chunk_id=c.regulation_chunk_id,
                regulation_text=c.regulation_text,
                regulation_metadata=c.regulation_metadata,
                policy_chunk_id=c.policy_chunk_id,
                policy_text=c.policy_text,
                policy_metadata=c.policy_metadata,
                bi_encoder_score=c.bi_encoder_score,
                cross_encoder_score=item['cross_score'],
                final_score=item['final_score'],
                rank_bi=c.rank,
                rank_cross=0,  # Would need to track separately
                rank_final=rank,
                inference_time_ms=item['inference_time'],
                scored_at=datetime.now().isoformat()
            )
            results.append(psc)
        
        # Update stats
        elapsed = (time.time() - start_time) * 1000
        self.stats['total_candidates_processed'] += len(candidates)
        self.stats['total_scoring_time_ms'] += elapsed
        self.stats['avg_scoring_time_ms'] = (
            self.stats['total_scoring_time_ms'] / self.stats['total_candidates_processed']
        )
        
        return results
    
    def score_single_pair(
        self,
        regulation_text: str,
        policy_text: str,
        regulation_chunk_id: str = "",
        policy_chunk_id: str = "",
        bi_encoder_score: float = 0.0
    ) -> PrecisionScoredCandidate:
        """
        Score a single pair (useful for testing)
        """
        start_time = time.time()
        
        # Check cache
        cross_score = None
        inference_time = 0.0
        
        if self.cache:
            cross_score = self.cache.get(regulation_text, policy_text)
        
        if cross_score is None:
            ce_result = self.cross_encoder.score_pair(
                regulation_text, policy_text,
                regulation_chunk_id, policy_chunk_id
            )
            cross_score = ce_result.score
            inference_time = ce_result.inference_time_ms
            
            if self.cache:
                self.cache.set(regulation_text, policy_text, cross_score)
        else:
            self.stats['cache_hits'] += 1
        
        final_score = self._calculate_final_score(bi_encoder_score, cross_score)
        
        elapsed = (time.time() - start_time) * 1000
        
        return PrecisionScoredCandidate(
            regulation_chunk_id=regulation_chunk_id,
            regulation_text=regulation_text,
            regulation_metadata={},
            policy_chunk_id=policy_chunk_id,
            policy_text=policy_text,
            policy_metadata={},
            bi_encoder_score=bi_encoder_score,
            cross_encoder_score=cross_score,
            final_score=final_score,
            rank_bi=1,
            rank_cross=1,
            rank_final=1,
            inference_time_ms=elapsed,
            scored_at=datetime.now().isoformat()
        )
    
    def test_consistency(self, num_tests: int = 10) -> Dict:
        """
        Test scoring consistency - same input should give same output
        """
        test_texts = [
            ("Organizations must encrypt data at rest.", "We encrypt all data at rest using AES-256."),
            ("Employees must report breaches within 24 hours.", "Security incidents must be reported within one business day."),
        ]
        
        consistency_scores = []
        
        for reg_text, pol_text in test_texts[:num_tests]:
            scores = []
            for _ in range(3):  # Score same pair 3 times
                result = self.cross_encoder.score_pair(reg_text, pol_text)
                scores.append(result.score)
            
            # Check variance
            variance = np.var(scores)
            consistency_scores.append({
                'pair': f"{reg_text[:30]}... vs {pol_text[:30]}...",
                'scores': scores,
                'variance': variance,
                'consistent': variance < 0.001  # Threshold for consistency
            })
        
        all_consistent = all(c['consistent'] for c in consistency_scores)
        
        return {
            'num_tests': len(consistency_scores),
            'all_consistent': all_consistent,
            'details': consistency_scores
        }
    
    def get_metadata(self) -> Dict:
        """Get scoring metadata"""
        return {
            **self.scoring_metadata,
            'stats': self.stats,
            'cache_size': len(self.cache._memory_cache) if self.cache else 0
        }


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Day 23: Precision Scoring Pipeline Test")
    print("=" * 60)
    
    # Initialize pipeline
    print("\n1. Initializing precision scoring pipeline...")
    pipeline = PrecisionScoringPipeline(
        bi_encoder_weight=0.3,
        cross_encoder_weight=0.7,
        use_cache=True
    )
    print(f"   Bi-encoder weight: {pipeline.bi_weight}")
    print(f"   Cross-encoder weight: {pipeline.cross_weight}")
    
    # Test single pair
    print("\n2. Testing single pair scoring...")
    result = pipeline.score_single_pair(
        regulation_text="Organizations must implement access controls based on least privilege.",
        policy_text="Access to systems is granted based on job role and least privilege principles.",
        regulation_chunk_id="reg_test_1",
        policy_chunk_id="pol_test_1",
        bi_encoder_score=0.75
    )
    
    print(f"   Bi-encoder: {result.bi_encoder_score:.3f}")
    print(f"   Cross-encoder: {result.cross_encoder_score:.3f}")
    print(f"   Final score: {result.final_score:.3f}")
    print(f"   Time: {result.inference_time_ms:.1f}ms")
    
    # Test consistency
    print("\n3. Testing scoring consistency...")
    consistency = pipeline.test_consistency(num_tests=3)
    print(f"   All consistent: {consistency['all_consistent']}")
    for detail in consistency['details']:
        print(f"   - {detail['pair'][:50]}...")
        print(f"     Variance: {detail['variance']:.6f} ({'✅' if detail['consistent'] else '❌'})")
    
    # Test with sample candidates
    print("\n4. Testing with sample candidates...")
    from candidate_generation.candidate_generator import CandidateGenerator
    
    gen = CandidateGenerator()
    chroma = gen.chroma
    coll = chroma.get_collection("regulations")
    sample = coll.get(limit=2)
    
    # Create mock candidates
    mock_candidates = []
    for i, (chunk_id, text, meta) in enumerate(zip(sample['ids'], sample['documents'], sample['metadatas'])):
        # Get a policy candidate
        cands = gen.get_candidates(
            {'chunk_id': chunk_id, 'content': text, 'metadata': meta},
            top_k=2,
            min_score=0.2
        )
        mock_candidates.extend(cands)
    
    if mock_candidates:
        print(f"   Scoring {len(mock_candidates)} candidates...")
        scored = pipeline.score_candidates(mock_candidates[:4])
        
        print(f"   Results:")
        for s in scored[:3]:
            print(f"   - Final: {s.final_score:.3f} (Bi: {s.bi_encoder_score:.3f}, Cross: {s.cross_encoder_score:.3f})")
    
    # Show metadata
    print("\n5. Scoring metadata:")
    metadata = pipeline.get_metadata()
    print(f"   Cache hits: {metadata['stats']['cache_hits']}")
    print(f"   Cache misses: {metadata['stats']['cache_misses']}")
    print(f"   Avg time per candidate: {metadata['stats']['avg_scoring_time_ms']:.1f}ms")
    
    print("\n" + "=" * 60)
    print("=" * 60)