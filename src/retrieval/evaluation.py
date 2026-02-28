"""Retrieval Evaluation - Metrics and testing"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Tuple
import json
import numpy as np
from collections import defaultdict
import logging

from retrieval.semantic_search import SemanticSearch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    def __init__(self, search_engine: SemanticSearch = None):
        self.search = search_engine or SemanticSearch()
        self.results = []
    
    def precision_at_k(self, relevant: List[str], retrieved: List[str], k: int) -> float:
        """Calculate Precision@K"""
        if k == 0:
            return 0.0
        retrieved_k = retrieved[:k]
        relevant_set = set(relevant)
        retrieved_relevant = [r for r in retrieved_k if r in relevant_set]
        return len(retrieved_relevant) / k
    
    def recall_at_k(self, relevant: List[str], retrieved: List[str], k: int) -> float:
        """Calculate Recall@K"""
        if not relevant:
            return 0.0
        retrieved_k = retrieved[:k]
        relevant_set = set(relevant)
        retrieved_relevant = [r for r in retrieved_k if r in relevant_set]
        return len(retrieved_relevant) / len(relevant)
    
    def mean_reciprocal_rank(self, relevant: List[str], retrieved: List[str]) -> float:
        """Calculate MRR"""
        relevant_set = set(relevant)
        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant_set:
                return 1.0 / i
        return 0.0
    
    def average_precision(self, relevant: List[str], retrieved: List[str]) -> float:
        """Calculate Average Precision"""
        if not relevant:
            return 0.0
        
        relevant_set = set(relevant)
        precisions = []
        
        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant_set:
                precisions.append(self.precision_at_k(relevant, retrieved, i))
        
        if not precisions:
            return 0.0
        
        return sum(precisions) / len(relevant)
    
    def evaluate_query(
        self,
        query: str,
        relevant_ids: List[str],
        top_k: int = 10
    ) -> Dict:
        """Evaluate single query"""
        results = self.search.search(query, top_k=top_k)
        retrieved_ids = [r['id'] for r in results]
        
        metrics = {
            'query': query,
            'relevant': relevant_ids,
            'retrieved': retrieved_ids,
            'precision@1': self.precision_at_k(relevant_ids, retrieved_ids, 1),
            'precision@3': self.precision_at_k(relevant_ids, retrieved_ids, 3),
            'precision@5': self.precision_at_k(relevant_ids, retrieved_ids, 5),
            'precision@10': self.precision_at_k(relevant_ids, retrieved_ids, 10),
            'recall@5': self.recall_at_k(relevant_ids, retrieved_ids, 5),
            'recall@10': self.recall_at_k(relevant_ids, retrieved_ids, 10),
            'mrr': self.mean_reciprocal_rank(relevant_ids, retrieved_ids),
            'ap': self.average_precision(relevant_ids, retrieved_ids),
            'num_results': len(results)
        }
        
        return metrics
    
    def evaluate_dataset(self, test_dataset: List[Dict]) -> Dict:
        """Evaluate full test dataset"""
        all_metrics = []
        
        for item in test_dataset:
            metrics = self.evaluate_query(
                item['query'],
                item['relevant_ids'],
                item.get('top_k', 10)
            )
            all_metrics.append(metrics)
        
        # Aggregate
        summary = {
            'num_queries': len(all_metrics),
            'mean_precision@1': np.mean([m['precision@1'] for m in all_metrics]),
            'mean_precision@3': np.mean([m['precision@3'] for m in all_metrics]),
            'mean_precision@5': np.mean([m['precision@5'] for m in all_metrics]),
            'mean_precision@10': np.mean([m['precision@10'] for m in all_metrics]),
            'mean_recall@5': np.mean([m['recall@5'] for m in all_metrics]),
            'mean_recall@10': np.mean([m['recall@10'] for m in all_metrics]),
            'mean_mrr': np.mean([m['mrr'] for m in all_metrics]),
            'mean_ap': np.mean([m['ap'] for m in all_metrics]),  # MAP
            'queries': all_metrics
        }
        
        self.results = all_metrics
        return summary
    
    def create_test_dataset_template(self, output_path: str = "data/test_queries.json"):
        """Create template for manual annotation"""
        template = [
            {
                "query": "financial reporting quarterly requirements",
                "relevant_ids": ["chunk_id_1", "chunk_id_2"],
                "description": "Find regulations about quarterly financial reporting",
                "difficulty": "easy"
            },
            {
                "query": "data breach notification timeline",
                "relevant_ids": [],
                "description": "How quickly must data breaches be reported",
                "difficulty": "medium"
            },
            {
                "query": "environmental impact assessment exemptions",
                "relevant_ids": [],
                "description": "Cases where EIA is not required",
                "difficulty": "hard"
            }
        ]
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(template, f, indent=2)
        
        logger.info(f"Test dataset template created: {output_path}")
        return template
    
    def analyze_errors(self, min_precision: float = 0.5) -> List[Dict]:
        """Analyze queries with poor performance"""
        errors = []
        for result in self.results:
            if result['precision@5'] < min_precision:
                errors.append({
                    'query': result['query'],
                    'precision@5': result['precision@5'],
                    'relevant_not_retrieved': list(set(result['relevant']) - set(result['retrieved'][:5])),
                    'retrieved_irrelevant': list(set(result['retrieved'][:5]) - set(result['relevant']))
                })
        return errors
    
    def export_report(self, summary: Dict, output_path: str = "outputs/retrieval_report.json"):
        """Export evaluation report"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Report exported: {output_path}")


def create_synthetic_test_data(search_engine: SemanticSearch, num_queries: int = 20):
    """Generate synthetic test data from existing documents"""
    # Get all documents
    collection = search_engine.chroma_manager.get_collection()
    all_docs = collection.get()
    
    if not all_docs['documents']:
        logger.error("No documents in collection")
        return []
    
    test_data = []
    
    # Generate queries from document excerpts
    for i in range(min(num_queries, len(all_docs['documents']))):
        doc_text = all_docs['documents'][i]
        doc_id = all_docs['ids'][i]
        
        # Create synthetic query (first 10 words)
        words = doc_text.split()
        if len(words) > 10:
            query = ' '.join(words[:10])
        else:
            query = doc_text[:100]
        
        test_data.append({
            'query': query,
            'relevant_ids': [doc_id],
            'description': f'Synthetic query for {doc_id}',
            'difficulty': 'easy'
        })
    
    return test_data


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Day 12: Retrieval Evaluation Test")
    print("=" * 60)
    
    # Initialize
    search = SemanticSearch()
    evaluator = RetrievalEvaluator(search)
    
    # Check if we have real test data
    test_file = Path("data/test_queries.json")
    
    if not test_file.exists():
        print("\n1. Creating test dataset template...")
        evaluator.create_test_dataset_template()
        
        print("\n2. Generating synthetic test data...")
        test_data = create_synthetic_test_data(search, num_queries=10)
    else:
        print("\n1. Loading test dataset...")
        with open(test_file) as f:
            test_data = json.load(f)
    
    if not test_data:
        print(" No test data available. Add some PDFs first.")
        sys.exit(0)
    
    print(f"\n3. Evaluating {len(test_data)} queries...")
    summary = evaluator.evaluate_dataset(test_data)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Queries evaluated: {summary['num_queries']}")
    print(f"Mean Precision@1:  {summary['mean_precision@1']:.3f}")
    print(f"Mean Precision@3:  {summary['mean_precision@3']:.3f}")
    print(f"Mean Precision@5:  {summary['mean_precision@5']:.3f}")
    print(f"Mean Precision@10: {summary['mean_precision@10']:.3f}")
    print(f"Mean Recall@5:     {summary['mean_recall@5']:.3f}")
    print(f"Mean Recall@10:    {summary['mean_recall@10']:.3f}")
    print(f"Mean Reciprocal Rank: {summary['mean_mrr']:.3f}")
    print(f"Mean Average Precision: {summary['mean_ap']:.3f}")
    
    # Error analysis
    print("\n4. Error analysis...")
    errors = evaluator.analyze_errors(min_precision=0.5)
    print(f"   Queries with precision@5 < 0.5: {len(errors)}")
    
    # Export
    evaluator.export_report(summary)
    
    print("\n" + "=" * 60)
    print("=" * 60)