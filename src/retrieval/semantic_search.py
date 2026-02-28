"""Semantic Search - Query interface with filtering and hybrid search"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Optional
import numpy as np
from rank_bm25 import BM25Okapi
import re
import logging

from embeddings.embedding_generator import EmbeddingGenerator
from embeddings.chroma_manager import ChromaManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticSearch:
    def __init__(
        self,
        chroma_path: str = "data/processed/chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "regulations",
        use_hybrid: bool = True,
        alpha: float = 0.7
    ):
        self.embedding_gen = EmbeddingGenerator(model_name=embedding_model)
        self.chroma_manager = ChromaManager(
            persist_directory=chroma_path,
            collection_name=collection_name
        )
        self.collection_name = collection_name
        self.use_hybrid = use_hybrid
        self.alpha = alpha
        
        self.bm25 = None
        self.corpus = []
        self.corpus_ids = []
    
    def search(self, query: str, top_k: int = 5, collection: str = None, 
               filters: Optional[Dict] = None, min_score: float = 0.0) -> List[Dict]:
        collection = collection or self.collection_name
        query_embedding = self.embedding_gen.encode(query)
        where_clause = self._build_where_clause(filters) if filters else None
        
        results = self.chroma_manager.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k * 2 if self.use_hybrid else top_k,
            where=where_clause,
            collection_name=collection
        )
        
        semantic_results = self._format_results(results)
        
        if self.use_hybrid:
            keyword_results = self._keyword_search(query, top_k=top_k * 2)
            semantic_results = self._fuse_results(semantic_results, keyword_results, top_k=top_k)
        
        filtered = [r for r in semantic_results if r['score'] >= min_score]
        return self._normalize_scores(filtered[:top_k])
    
    def _build_where_clause(self, filters: Dict) -> Dict:
        where = {}
        for key, value in filters.items():
            if isinstance(value, dict):
                where[key] = value
            else:
                where[key] = value
        return where
    
    def _format_results(self, chroma_results: Dict) -> List[Dict]:
        formatted = []
        if not chroma_results['ids']:
            return formatted
        
        ids = chroma_results['ids'][0]
        documents = chroma_results['documents'][0]
        metadatas = chroma_results['metadatas'][0]
        distances = chroma_results.get('distances', [[]])[0]
        
        for idx, (id_, doc, meta, dist) in enumerate(zip(ids, documents, metadatas, distances)):
            similarity = 1 - (dist if dist else 0)
            result = {
                'id': id_,
                'content': doc,
                'metadata': meta,
                'score': similarity,
                'source': 'semantic'
            }
            formatted.append(result)
        return formatted
    
    def _build_bm25_index(self, documents: List[str], ids: List[str]):
        tokenized = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        self.corpus = documents
        self.corpus_ids = ids
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())
    
    def _keyword_search(self, query: str, top_k: int = 5) -> List[Dict]:
        collection = self.chroma_manager.get_collection(self.collection_name)
        all_docs = collection.get()
        
        if not all_docs['documents']:
            return []
        
        if self.bm25 is None or len(self.corpus) != len(all_docs['documents']):
            self._build_bm25_index(all_docs['documents'], all_docs['ids'])
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    'id': self.corpus_ids[idx],
                    'content': self.corpus[idx],
                    'score': float(scores[idx]),
                    'source': 'keyword'
                })
        return results
    
    def _fuse_results(self, semantic_results: List[Dict], keyword_results: List[Dict], top_k: int) -> List[Dict]:
        def normalize(results: List[Dict]) -> List[Dict]:
            if not results:
                return []
            max_score = max(r['score'] for r in results)
            if max_score == 0:
                return results
            for r in results:
                r['score'] = r['score'] / max_score
            return results
        
        semantic_results = normalize(semantic_results)
        keyword_results = normalize(keyword_results)
        
        combined = {}
        for r in semantic_results:
            combined[r['id']] = {**r, 'semantic_score': r['score'], 'keyword_score': 0, 'score': r['score'] * self.alpha}
        
        for r in keyword_results:
            if r['id'] in combined:
                combined[r['id']]['keyword_score'] = r['score']
                combined[r['id']]['score'] += r['score'] * (1 - self.alpha)
                combined[r['id']]['source'] = 'hybrid'
            else:
                combined[r['id']] = {**r, 'semantic_score': 0, 'keyword_score': r['score'], 'score': r['score'] * (1 - self.alpha)}
        
        fused = sorted(combined.values(), key=lambda x: x['score'], reverse=True)
        return fused[:top_k]
    
    def _normalize_scores(self, results: List[Dict]) -> List[Dict]:
        if not results:
            return results
        
        scores = [r['score'] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            for r in results:
                r['score'] = 1.0
        else:
            for r in results:
                r['score'] = (r['score'] - min_score) / (max_score - min_score)
        return results
    
    def get_search_explanation(self, result: Dict) -> str:
        """Generate human-readable explanation of why result was retrieved"""
        explanation = []
        
        source_type = result.get('source', 'unknown')
        if source_type == 'semantic':
            explanation.append("Retrieved by semantic similarity to query meaning.")
        elif source_type == 'keyword':
            explanation.append("Retrieved by keyword matching.")
        else:
            explanation.append("Retrieved by combined semantic and keyword matching.")
        
        meta = result.get('metadata', {})
        if meta.get('page_number'):
            explanation.append(f"From page {meta['page_number']} of {meta.get('source', 'unknown document')}.")
        
        return " ".join(explanation)
    
    def search_by_page_range(self, query: str, source: str, start_page: int, end_page: int, top_k: int = 5) -> List[Dict]:
        filters = {"source": source}
        results = self.search(query, top_k=top_k * 3, filters=filters)
        
        filtered_results = []
        for result in results:
            page = result.get('metadata', {}).get('page_number')
            if page is not None and start_page <= page <= end_page:
                filtered_results.append(result)
        return filtered_results[:top_k]


# TEST
if __name__ == "__main__":
    print("=" * 60)
    print("Day 11: Semantic Search Test")
    print("=" * 60)
    
    search = SemanticSearch(collection_name="regulations", use_hybrid=True)
    
    test_queries = [
        "suspicious transaction reporting",
        "data protection",
        "customer identification"
    ]
    
    print("\nTesting search...")
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = search.search(query, top_k=3)
        for i, res in enumerate(results, 1):
            print(f"  {i}. [{res['score']:.3f}] {res['content'][:60]}...")
            print(f"     Source: {res.get('metadata', {}).get('source', 'unknown')}")
    
    print("\n Day 11 complete!")