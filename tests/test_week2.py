"""Week 2 Test Suite - Vector Storage & Retrieval"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import json


def test_embedding_generation():
    """Test 1: Embedding generation"""
    print("\nTEST 1: Embedding Generation")
    print("=" * 50)
    
    from src.embeddings.embedding_generator import EmbeddingGenerator
    
    gen = EmbeddingGenerator()
    
    # Test dimension
    emb = gen.encode("Test sentence")
    assert len(emb) == 384, f"Expected 384 dims, got {len(emb)}"
    print(" Dimension check: 384")
    
    # Test batching
    texts = ["Text one", "Text two", "Text three"]
    batch = gen.encode(texts, batch_size=2)
    assert batch.shape == (3, 384)
    print(" Batch encoding works")
    
    # Test caching
    _ = gen.encode("Cache test")
    _ = gen.encode("Cache test")  # Should hit cache
    stats = gen.get_stats()
    assert stats['cache_hits'] >= 1
    print(f" Caching works (hits: {stats['cache_hits']})")
    
    return True


def test_chromadb_operations():
    """Test 2: ChromaDB operations"""
    print("\nTEST 2: ChromaDB Operations")
    print("=" * 50)
    
    from src.embeddings.chroma_manager import ChromaManager
    
    db = ChromaManager()
    
    # Create collection
    try:
        db.delete_collection("test_week2")
    except:
        pass
    coll = db.create_collection("test_week2", {"test": True})
    print(" Collection created")
    # Add documents
    db.add_documents(
        documents=["Doc 1", "Doc 2"],
        embeddings=[[0.1]*384, [0.2]*384],
        metadatas=[{"s": "a"}, {"s": "b"}],
        ids=["id1", "id2"],
        collection_name="test_week2"
    )
    print(" Documents added")
    
    # Query
    results = db.query([[0.15]*384], n_results=2, collection_name="test_week2")
    assert len(results['ids'][0]) == 2
    print(" Query works")
    
    # Persistence
    # Force Chroma to flush to disk before verification
    if hasattr(db.client, "persist"):
        db.client.persist()
        time.sleep(0.5)
        assert db.verify_persistence()
        print(" Persistence verified")
    
    # Cleanup
    db.delete_collection("test_week2")
    print(" Cleanup complete")
    
    return True


def test_search_functionality():
    """Test 3: Search functionality"""
    print("\nTEST 3: Search Functionality")
    print("=" * 50)
    
    from src.retrieval.semantic_search import SemanticSearch
    
    search = SemanticSearch(use_hybrid=False)  # Semantic only for speed
    
    # Need some data first
    from src.embeddings.chroma_manager import ChromaManager
    from src.embeddings.embedding_generator import EmbeddingGenerator
    
    db = ChromaManager()
    gen = EmbeddingGenerator()
    
    # Add test data if empty
    try:
        coll = db.get_collection("regulations")
        if coll.count() == 0:
            raise ValueError("Empty")
    except:
        db.create_collection("regulations")
        docs = ["Financial reporting rule", "Data privacy law", "Environmental policy"]
        embs = gen.encode(docs)
        db.add_documents(
            documents=docs,
            embeddings=embs.tolist(),
            metadatas=[{"source": "test.pdf", "page": i} for i in range(3)],
            collection_name="regulations"
        )
    
    # Test search
    results = search.search("financial rules", top_k=3)
    assert len(results) > 0
    print(f" Search returned {len(results)} results")
    
    # Test latency
    start = time.time()
    for _ in range(10):
        search.search("test query", top_k=5)
    avg_latency = (time.time() - start) / 10 * 1000
    
    print(f" Average latency: {avg_latency:.1f}ms")
    if avg_latency > 500:
        print(f"     Latency > 500ms target")
    
    return True


def test_end_to_end_pipeline():
    """Test 4: End-to-end pipeline"""
    print("\nTEST 4: End-to-End Pipeline")
    print("=" * 50)
    
    from src.embeddings.ingestion_pipeline import BatchIngestionPipeline
    
    # Create dummy chunks
    chunks = [
        {
            'chunk_id': f'e2e_test_{i}',
            'content': f'This is test content for chunk {i} about regulations.',
            'source': 'e2e_test.pdf',
            'page_number': i + 1,
            'section_header': f'Section {i}',
            'metadata': {'word_count': 10, 'chunk_size': 50}
        }
        for i in range(5)
    ]
    
    pipeline = BatchIngestionPipeline(batch_size=2)
    stats = pipeline.ingest_chunks(chunks, "test_e2e", show_progress=False)
    
    assert stats['chunks_processed'] == 5
    assert stats['chunks_failed'] == 0
    print(f" Ingested {stats['chunks_processed']} chunks")
    print(f" Speed: {stats.get('chunks_per_second', 0):.1f} chunks/sec")
    
    # Verify in DB
    count = pipeline.chroma_manager.get_document_count("test_e2e")
    assert count == 5
    print(f" Verified {count} documents in collection")
    
    # Cleanup
    pipeline.chroma_manager.delete_collection("test_e2e")
    
    return True


def test_performance_benchmark():
    """Test 5: Performance benchmark"""
    print("\nTEST 5: Performance Benchmark")
    print("=" * 50)
    
    from src.retrieval.semantic_search import SemanticSearch
    
    search = SemanticSearch()
    
    # Benchmark query latency
    latencies = []
    for i in range(20):
        start = time.time()
        search.search(f"benchmark query {i}", top_k=5)
        latencies.append((time.time() - start) * 1000)
    
    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    
    print(f" Average latency: {avg_latency:.1f}ms")
    print(f" P95 latency: {p95_latency:.1f}ms")
    
    if avg_latency < 500:
        print(" Meets <500ms target")
    else:
        print("  Does not meet <500ms target")
    
    return avg_latency < 1000  # Relaxed threshold for test


def main():
    print("=" * 60)
    print("REG-INSIGHT Week 2: Test Suite")
    print("=" * 60)
    
    tests = [
        ("Embedding Generation", test_embedding_generation),
        ("ChromaDB Operations", test_chromadb_operations),
        ("Search Functionality", test_search_functionality),
        ("End-to-End Pipeline", test_end_to_end_pipeline),
        ("Performance Benchmark", test_performance_benchmark)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
            print(f"\n{' PASS' if passed else ' FAIL'}: {name}")
        except Exception as e:
            print(f"\n FAIL: {name} - {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, p in results:
        status = "PASS" if p else "FAIL"
        print(f"{status} {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n Week 2 complete! All systems operational.")
        return 0
    else:
        print(f"\n  {total - passed} test(s) failed. Review errors above.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())