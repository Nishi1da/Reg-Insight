"""Cross-Encoder Scorer - Precise pairwise similarity scoring"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import logging
import time
from dataclasses import dataclass
from functools import lru_cache

from sentence_transformers import CrossEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CrossEncoderScore:
    """Represents a cross-encoder scoring result"""
    regulation_chunk_id: str
    policy_chunk_id: str
    pair_text: Tuple[str, str]  # (regulation_text, policy_text)
    score: float  # 0-1 range
    inference_time_ms: float
    model_version: str


class CrossEncoderScorer:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 16
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size
        
        logger.info(f"Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(
            model_name,
            device=self.device,
            max_length=max_length
        )
        
        self.stats = {
            'total_pairs_scored': 0,
            'total_inference_time_ms': 0.0,
            'avg_inference_time_ms': 0.0,
            'batch_calls': 0
        }
        
        # Simple in-memory cache for identical pairs
        self._cache: Dict[str, float] = {}
    
    def _make_cache_key(self, reg_text: str, policy_text: str) -> str:
        """Create cache key from text pair"""
        # Simple hash of combined text
        return f"{hash(reg_text[:100])}_{hash(policy_text[:100])}"
    
    def score_pair(
        self,
        regulation_text: str,
        policy_text: str,
        regulation_chunk_id: str = "",
        policy_chunk_id: str = ""
    ) -> CrossEncoderScore:
        """
        Score a single regulation-policy pair
        
        Args:
            regulation_text: Regulation chunk text
            policy_text: Policy chunk text
            regulation_chunk_id: ID for tracking
            policy_chunk_id: ID for tracking
        
        Returns:
            CrossEncoderScore with score 0-1
        """
        start_time = time.time()
        
        # Check cache
        cache_key = self._make_cache_key(regulation_text, policy_text)
        if cache_key in self._cache:
            score = self._cache[cache_key]
            inference_time = 0.0  # Cache hit
        else:
            # Prepare input
            pair = [regulation_text, policy_text]
            
            # Run inference
            with torch.no_grad():
                raw_score = self.model.predict([pair], show_progress_bar=False)[0]
            
            # Convert to 0-1 range (ms-marco outputs raw logits)
            score = float(torch.sigmoid(torch.tensor(raw_score)))
            
            # Store in cache
            self._cache[cache_key] = score
            
            inference_time = (time.time() - start_time) * 1000
        
        # Update stats
        self.stats['total_pairs_scored'] += 1
        self.stats['total_inference_time_ms'] += inference_time
        self.stats['avg_inference_time_ms'] = (
            self.stats['total_inference_time_ms'] / self.stats['total_pairs_scored']
        )
        
        return CrossEncoderScore(
            regulation_chunk_id=regulation_chunk_id,
            policy_chunk_id=policy_chunk_id,
            pair_text=(regulation_text[:100], policy_text[:100]),
            score=score,
            inference_time_ms=inference_time,
            model_version=self.model_name
        )
    
    def score_batch(
        self,
        pairs: List[Tuple[str, str, str, str]],  # (reg_id, reg_text, pol_id, pol_text)
        show_progress: bool = True
    ) -> List[CrossEncoderScore]:
        """
        Score multiple pairs in batches for efficiency
        
        Args:
            pairs: List of (reg_id, reg_text, pol_id, pol_text)
            show_progress: Show progress bar
        
        Returns:
            List of CrossEncoderScore
        """
        if not pairs:
            return []
        
        results = []
        total_batches = (len(pairs) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(pairs))
            batch = pairs[start_idx:end_idx]
            
            batch_start = time.time()
            
            # Prepare texts for model
            batch_texts = [[p[1], p[3]] for p in batch]  # (reg_text, pol_text)
            
            # Run batch inference
            with torch.no_grad():
                raw_scores = self.model.predict(
                    batch_texts,
                    show_progress_bar=False,
                    batch_size=len(batch_texts)
                )
            
            # Convert scores
            batch_time = (time.time() - batch_start) * 1000
            
            for i, (reg_id, reg_text, pol_id, pol_text) in enumerate(batch):
                score = float(torch.sigmoid(torch.tensor(raw_scores[i])))
                
                results.append(CrossEncoderScore(
                    regulation_chunk_id=reg_id,
                    policy_chunk_id=pol_id,
                    pair_text=(reg_text[:100], pol_text[:100]),
                    score=score,
                    inference_time_ms=batch_time / len(batch),
                    model_version=self.model_name
                ))
            
            self.stats['batch_calls'] += 1
            
            if show_progress and (batch_idx + 1) % 10 == 0:
                logger.info(f"  Processed batch {batch_idx + 1}/{total_batches}")
        
        # Update stats
        self.stats['total_pairs_scored'] += len(pairs)
        
        return results
    
    def compare_with_bi_encoder(
        self,
        candidates: List,  # CandidatePair objects
        bi_encoder_scores: List[float]
    ) -> Dict:
        """
        Compare bi-encoder and cross-encoder scores
        
        Args:
            candidates: List of CandidatePair
            bi_encoder_scores: Corresponding bi-encoder scores
        
        Returns:
            Comparison statistics
        """
        # Score with cross-encoder
        pairs = []
        for c in candidates:
            pairs.append((
                c.regulation_chunk_id,
                c.regulation_text,
                c.policy_chunk_id,
                c.policy_text
            ))
        
        ce_results = self.score_batch(pairs, show_progress=False)
        ce_scores = [r.score for r in ce_results]
        
        # Calculate correlations and differences
        bi_scores = np.array(bi_encoder_scores)
        cross_scores = np.array(ce_scores)
        
        # Pearson correlation
        correlation = np.corrcoef(bi_scores, cross_scores)[0, 1]
        
        # Score differences
        differences = cross_scores - bi_scores
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)
        
        # Rank changes
        bi_ranks = np.argsort(-bi_scores)
        ce_ranks = np.argsort(-cross_scores)
        
        rank_changes = []
        for i in range(len(candidates)):
            bi_rank = np.where(bi_ranks == i)[0][0]
            ce_rank = np.where(ce_ranks == i)[0][0]
            rank_changes.append(abs(bi_rank - ce_rank))
        
        return {
            'num_pairs': len(candidates),
            'pearson_correlation': float(correlation),
            'mean_score_difference': float(mean_diff),
            'std_score_difference': float(std_diff),
            'bi_encoder_mean': float(np.mean(bi_scores)),
            'cross_encoder_mean': float(np.mean(cross_scores)),
            'max_rank_change': int(max(rank_changes)),
            'avg_rank_change': float(np.mean(rank_changes)),
            'cross_encoder_faster': False,  # Cross-encoder is slower but more accurate
            'sample_comparisons': [
                {
                    'reg_id': c.regulation_chunk_id,
                    'pol_id': c.policy_chunk_id,
                    'bi_encoder': float(b),
                    'cross_encoder': float(ce),
                    'difference': float(ce - b)
                }
                for c, b, ce in zip(candidates[:5], bi_scores[:5], cross_scores[:5])
            ]
        }
    
    def get_stats(self) -> Dict:
        """Get scoring statistics"""
        return self.stats.copy()
    
    def clear_cache(self):
        """Clear scoring cache"""
        self._cache.clear()
        logger.info("Cache cleared")


# Test
if __name__ == "__main__":
    print("=" * 60)
    print(" Cross-Encoder Scorer Test")
    print("=" * 60)
    
    # Initialize
    print("\n1. Loading cross-encoder model...")
    scorer = CrossEncoderScorer()
    print(f"   Device: {scorer.device}")
    print(f"   Model: {scorer.model_name}")
    
    # Test single scoring
    print("\n2. Testing single pair scoring...")
    reg_text = "Organizations must implement multi-factor authentication for all remote access to sensitive systems."
    policy_text = "All employees must use MFA when accessing company resources remotely via VPN."
    
    result = scorer.score_pair(
        reg_text, policy_text,
        regulation_chunk_id="reg_001",
        policy_chunk_id="pol_042"
    )
    
    print(f"   Score: {result.score:.3f}")
    print(f"   Time: {result.inference_time_ms:.1f}ms")
    print(f"   Model: {result.model_version}")
    
    # Test batch scoring
    print("\n3. Testing batch scoring (32 pairs)...")
    test_pairs = []
    for i in range(32):
        test_pairs.append((
            f"reg_{i}",
            f"Regulation requirement {i}: Organizations must maintain records of processing activities.",
            f"pol_{i}",
            f"Policy section {i}: We document all data processing operations in our compliance register."
        ))
    
    batch_results = scorer.score_batch(test_pairs)
    print(f"   Scored {len(batch_results)} pairs")
    print(f"   Score range: {min(r.score for r in batch_results):.3f} - {max(r.score for r in batch_results):.3f}")
    print(f"   Avg time per pair: {scorer.stats['avg_inference_time_ms']:.1f}ms")
    
    # Test cache
    print("\n4. Testing cache (scoring same pair again)...")
    cached_result = scorer.score_pair(reg_text, policy_text)
    print(f"   Cached time: {cached_result.inference_time_ms:.1f}ms (should be 0.0)")
    
    # Show stats
    print("\n5. Final statistics:")
    stats = scorer.get_stats()
    print(f"   Total pairs scored: {stats['total_pairs_scored']}")
    print(f"   Batch calls: {stats['batch_calls']}")
    print(f"   Cache size: {len(scorer._cache)}")
    
    print("\n" + "=" * 60)
    print("=" * 60)