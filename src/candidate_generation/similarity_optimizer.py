"""Similarity Optimizer - Efficient batch similarity computation"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Tuple
import numpy as np
import time
import logging
from functools import lru_cache

from embeddings.embedding_generator import EmbeddingGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimilarityOptimizer:
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.stats = {
            'batch_computations': 0,
            'avg_batch_time_ms': 0.0
        }
    
    def cosine_similarity_batch(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Efficient batch cosine similarity using matrix operations
        
        Args:
            query_embedding: (384,) array
            candidate_embeddings: (N, 384) array
        
        Returns:
            similarities: (N,) array of scores 0-1
        """
        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Normalize candidates (row-wise)
        candidate_norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-8
        candidates_normalized = candidate_embeddings / candidate_norms
        
        # Batch dot product
        similarities = np.dot(candidates_normalized, query_norm)
        
        return similarities
    
    def compute_similarity_matrix(
        self,
        reg_embeddings: np.ndarray,
        policy_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute full similarity matrix between regulations and policies
        
        Args:
            reg_embeddings: (N_reg, 384)
            policy_embeddings: (N_policy, 384)
        
        Returns:
            similarity_matrix: (N_reg, N_policy)
        """
        start = time.time()
        
        # Normalize both sets
        reg_norms = np.linalg.norm(reg_embeddings, axis=1, keepdims=True) + 1e-8
        reg_normalized = reg_embeddings / reg_norms
        
        policy_norms = np.linalg.norm(policy_embeddings, axis=1, keepdims=True) + 1e-8
        policy_normalized = policy_embeddings / policy_norms
        
        # Matrix multiplication: (N_reg, 384) @ (384, N_policy) = (N_reg, N_policy)
        similarity_matrix = np.dot(reg_normalized, policy_normalized.T)
        
        elapsed = (time.time() - start) * 1000  # ms
        
        self.stats['batch_computations'] += 1
        self.stats['avg_batch_time_ms'] = (
            (self.stats['avg_batch_time_ms'] * (self.stats['batch_computations'] - 1) + elapsed)
            / self.stats['batch_computations']
        )
        
        logger.info(f"Computed {reg_embeddings.shape[0]}x{policy_embeddings.shape[0]} matrix in {elapsed:.1f}ms")
        
        return similarity_matrix
    
    def analyze_score_distribution(self, scores: np.ndarray) -> Dict:
        """
        Analyze distribution of similarity scores for threshold tuning
        
        Args:
            scores: Array of similarity scores
        
        Returns:
            Statistics dict
        """
        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'median': float(np.median(scores)),
            'percentile_25': float(np.percentile(scores, 25)),
            'percentile_75': float(np.percentile(scores, 75)),
            'percentile_90': float(np.percentile(scores, 90)),
            'above_0_7': int(np.sum(scores >= 0.7)),
            'above_0_5': int(np.sum(scores >= 0.5)),
            'above_0_3': int(np.sum(scores >= 0.3))
        }
    
    def suggest_threshold(self, scores: np.ndarray, target_percentile: int = 75) -> float:
        """
        Suggest dynamic threshold based on score distribution
        
        Args:
            scores: Similarity scores
            target_percentile: Keep top X% as candidates
        
        Returns:
            Suggested threshold
        """
        return float(np.percentile(scores, 100 - target_percentile))
    
    def benchmark_speed(self, n_regs: int = 100, n_policies: int = 530) -> Dict:
        """
        Benchmark similarity computation speed
        
        Args:
            n_regs: Number of regulation chunks
            n_policies: Number of policy chunks
        
        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking: {n_regs} regs x {n_policies} policies...")
        
        # Generate random embeddings
        reg_emb = np.random.randn(n_regs, self.embedding_dim).astype(np.float32)
        policy_emb = np.random.randn(n_policies, self.embedding_dim).astype(np.float32)
        
        # Warmup
        _ = self.compute_similarity_matrix(reg_emb[:10], policy_emb[:10])
        
        # Benchmark
        times = []
        for _ in range(5):
            start = time.time()
            sim_matrix = self.compute_similarity_matrix(reg_emb, policy_emb)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
        
        avg_time = np.mean(times)
        per_regulation = avg_time / n_regs
        
        logger.info(f"Average time: {avg_time:.1f}ms ({per_regulation:.2f}ms per regulation)")
        
        # Check if meets target
        meets_target = per_regulation < 100  # <100ms target
        
        return {
            'avg_time_ms': float(avg_time),
            'per_regulation_ms': float(per_regulation),
            'meets_target': meets_target,
            'matrix_shape': sim_matrix.shape
        }


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Day 16: Similarity Optimizer Test")
    print("=" * 60)
    
    opt = SimilarityOptimizer()
    
    # Test 1: Batch similarity
    print("\n1. Testing batch cosine similarity...")
    query = np.random.randn(384).astype(np.float32)
    candidates = np.random.randn(100, 384).astype(np.float32)
    
    scores = opt.cosine_similarity_batch(query, candidates)
    print(f"   Scores shape: {scores.shape}")
    print(f"   Score range: {scores.min():.3f} to {scores.max():.3f}")
    
    # Test 2: Similarity matrix
    print("\n2. Testing similarity matrix computation...")
    reg_emb = np.random.randn(50, 384).astype(np.float32)
    policy_emb = np.random.randn(200, 384).astype(np.float32)
    
    sim_matrix = opt.compute_similarity_matrix(reg_emb, policy_emb)
    print(f"   Matrix shape: {sim_matrix.shape}")
    
    # Test 3: Distribution analysis
    print("\n3. Analyzing score distribution...")
    stats = opt.analyze_score_distribution(sim_matrix.flatten())
    print(f"   Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
    print(f"   Above 0.7: {stats['above_0_7']}, Above 0.3: {stats['above_0_3']}")
    
    # Test 4: Threshold suggestion
    threshold = opt.suggest_threshold(sim_matrix.flatten(), target_percentile=75)
    print(f"\n4. Suggested threshold (top 25%): {threshold:.3f}")
    
    # Test 5: Speed benchmark
    print("\n5. Running speed benchmark...")
    bench = opt.benchmark_speed(n_regs=100, n_policies=530)
    print(f"   Per-regulation time: {bench['per_regulation_ms']:.2f}ms")
    print(f"   Meets <100ms target: {'PASS' if bench['meets_target'] else 'FAIL'}")
    
    print("\n" + "=" * 60)
    print("=" * 60)