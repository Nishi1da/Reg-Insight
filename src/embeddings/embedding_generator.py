"""Embedding Generator - Sentence Transformers with Caching"""

from sentence_transformers import SentenceTransformer
import numpy as np
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: str = "data/processed/embeddings_cache",
        device: str = None,
        use_cache: bool = True
    ):
        self.model_name = model_name
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if use_cache else None
        
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        logger.info(f"Loading model: {model_name}")
        self.device = device or ("cuda" if self._check_cuda() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Verify dimensions
        test_embedding = self.model.encode("test")
        self.embedding_dim = len(test_embedding)
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        
        # Cache tracking
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path"""
        return self.cache_dir / f"{key}.npy"
    
    def _load_from_cache(self, text: str) -> Union[np.ndarray, None]:
        """Load embedding from cache if exists"""
        if not self.use_cache or self.cache_dir is None:
            return None
            
        key = self._get_cache_key(text)
        cache_path = self._get_cache_path(key)
        
        if cache_path.exists():
            self.cache_hits += 1
            return np.load(cache_path)
        
        self.cache_misses += 1
        return None
    
    def _save_to_cache(self, text: str, embedding: np.ndarray):
        """Save embedding to cache"""
        if not self.use_cache or self.cache_dir is None:
            return
            
        key = self._get_cache_key(text)
        cache_path = self._get_cache_path(key)
        np.save(cache_path, embedding)
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings with caching and batching
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            use_cache: Use disk cache
        
        Returns:
            numpy array of embeddings (N, 384)
        """
        # Handle single text
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        
        embeddings = []
        texts_to_encode = []
        indices_to_encode = []
        
        # Check cache first (respect both instance and method parameter)
        should_use_cache = self.use_cache and use_cache
        
        if should_use_cache:
            for idx, text in enumerate(texts):
                cached = self._load_from_cache(text)
                if cached is not None:
                    embeddings.append((idx, cached))
                else:
                    texts_to_encode.append(text)
                    indices_to_encode.append(idx)
        else:
            texts_to_encode = texts
            indices_to_encode = list(range(len(texts)))
        
        # Encode non-cached texts
        if texts_to_encode:
            logger.info(f"Encoding {len(texts_to_encode)} texts (batch_size={batch_size})")
            
            new_embeddings = self.model.encode(
                texts_to_encode,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            # Save to cache and store
            for idx, text, emb in zip(indices_to_encode, texts_to_encode, new_embeddings):
                if should_use_cache:
                    self._save_to_cache(text, emb)
                embeddings.append((idx, emb))
        
        # Sort by original index
        embeddings.sort(key=lambda x: x[0])
        result = np.stack([emb for _, emb in embeddings])
        
        if single_input:
            result = result[0]
        
        return result
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "embedding_dim": self.embedding_dim,
            "device": self.device
        }
    
    def verify_dimension(self, expected_dim: int = 384) -> bool:
        """Verify embedding dimension"""
        test_text = "This is a test sentence for dimension verification."
        embedding = self.encode(test_text)
        
        actual_dim = len(embedding)
        is_correct = actual_dim == expected_dim
        
        if is_correct:
            logger.info(f"✅ Dimension check passed: {actual_dim}")
        else:
            logger.error(f"❌ Dimension mismatch: expected {expected_dim}, got {actual_dim}")
        
        return is_correct
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts"""
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        return float(similarity)


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Day 8: Embedding Generator Test")
    print("=" * 60)
    
    # Initialize
    generator = EmbeddingGenerator()
    
    # Test 1: Dimension verification
    print("\n1. Verifying embedding dimensions...")
    assert generator.verify_dimension(384), "Dimension check failed!"
    
    # Test 2: Single text encoding
    print("\n2. Testing single text encoding...")
    text = "This is a sample regulation about financial compliance."
    embedding = generator.encode(text)
    print(f"   Input: {text[:50]}...")
    print(f"   Output shape: {embedding.shape}")
    print(f"   Sample values: {embedding[:3]}")
    
    # Test 3: Batch encoding
    print("\n3. Testing batch encoding...")
    texts = [
        "Financial regulations require quarterly reporting.",
        "Data privacy laws protect consumer information.",
        "Environmental policies mandate emission controls.",
        "Labor laws specify minimum wage requirements."
    ]
    batch_embeddings = generator.encode(texts, batch_size=2, show_progress=True)
    print(f"   Batch shape: {batch_embeddings.shape}")
    
    # Test 4: Caching
    print("\n4. Testing cache functionality...")
    # First call (cache miss)
    _ = generator.encode("Cache test sentence")
    # Second call (cache hit)
    _ = generator.encode("Cache test sentence")
    print(f"   Cache stats: {generator.get_stats()}")
    
    # Test 5: Similarity
    print("\n5. Testing similarity computation...")
    pairs = [
        ("Financial reporting is required quarterly.", "Quarterly financial reports are mandatory."),
        ("Data privacy protects users.", "Environmental laws protect nature."),
    ]
    
    for text1, text2 in pairs:
        sim = generator.compute_similarity(text1, text2)
        print(f"   Similarity: {sim:.3f}")
        print(f"      '{text1[:40]}...'")
        print(f"      '{text2[:40]}...'")
        print()
    
    print("=" * 60)
    print("✅ Day 8 complete!")
    print("=" * 60)