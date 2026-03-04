"""Candidate Generation Pipeline - Full integration"""

import json  
from datetime import datetime
import logging
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Optional

from candidate_generation.candidate_generator import CandidateGenerator, CandidatePair
from candidate_generation.similarity_optimizer import SimilarityOptimizer
from candidate_generation.ranker import CandidateRanker, RankedCandidate
from candidate_generation.edge_case_handler import EdgeCaseHandler, EdgeCaseResult
from candidate_generation.retrieval_logger import RetrievalLogger
from candidate_generation.validator import CandidateValidator

_module_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CandidateGenerationPipeline:
    """
    End-to-end candidate generation pipeline
    
    Combines all Week 3 components:
    - Candidate generation (Day 15)
    - Similarity optimization (Day 16)
    - Ranking (Day 17)
    - Edge case handling (Day 18)
    - Logging (Day 19)
    - Validation (Day 20)
    """
    
    def __init__(
        self,
        collection_name: str = "regulations",
        top_k: int = 3,
        min_score: float = 0.3,
        similarity_weight: float = 0.7,
        section_weight: float = 0.3
    ):
        self.collection_name = collection_name
        self.top_k = top_k
        self.min_score = min_score
        
        # Initialize components
        self.generator = CandidateGenerator(policy_collection=collection_name)
        self.optimizer = SimilarityOptimizer()
        self.ranker = CandidateRanker(similarity_weight, section_weight)
        self.edge_handler = EdgeCaseHandler(no_match_threshold=min_score)
        self.logger = RetrievalLogger()
        
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_regulations': 0,
            'total_candidates': 0,
            'edge_cases': {
                'no_match': 0,
                'ambiguous': 0,
                'low_confidence': 0,
                'clear_match': 0
            }
        }
    
    def process_regulation_chunk(
        self,
        regulation_chunk: Dict
    ) -> Dict:
        """
        Process single regulation chunk through full pipeline
        
        Returns:
            Dict with candidates, ranking, and edge case info
        """
        start_time = datetime.now()
        
        # Step 1: Generate candidates
        raw_candidates = self.generator.get_candidates(
            regulation_chunk,
            top_k=self.top_k * 2,  # Get extra for ranking
            min_score=self.min_score
        )
        
        # Step 2: Rank candidates
        ranked_candidates = self.ranker.rank_candidates(raw_candidates)
        
        # Step 3: Deduplicate
        unique_candidates = self.ranker.deduplicate_candidates(ranked_candidates)
        
        # Step 4: Handle edge cases
        edge_result = self.edge_handler.classify_case(
            regulation_chunk['chunk_id'],
            unique_candidates[:self.top_k]
        )
        
        # Step 5: Log
        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
        self.logger.log_retrieval(
            regulation_chunk['chunk_id'],
            unique_candidates[:self.top_k],
            elapsed_ms,
            {'top_k': self.top_k, 'min_score': self.min_score}
        )
        
        # Update stats
        self.stats['edge_cases'][edge_result.classification.replace(' ', '_')] = \
            self.stats['edge_cases'].get(edge_result.classification.replace(' ', '_'), 0) + 1
        
        return {
            'regulation_chunk_id': regulation_chunk['chunk_id'],
            'candidates': unique_candidates[:self.top_k],
            'edge_case': edge_result,
            'processing_time_ms': elapsed_ms
        }
    
    def process_all(
        self,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Process all regulation chunks
        
        Args:
            limit: Process only first N chunks (for testing)
        
        Returns:
            List of results
        """
        self.stats['start_time'] = datetime.now()
        
        # Get all regulation chunks
        collection = self.generator.chroma.get_collection(self.collection_name)
        all_data = collection.get(limit=limit)
        
        self.stats['total_regulations'] = len(all_data['ids'])
        
        _module_logger.info(f"Processing {len(all_data['ids'])} regulation chunks...")
        
        results = []
        
        for i, (chunk_id, text, metadata) in enumerate(tqdm(
            zip(all_data['ids'], all_data['documents'], all_data['metadatas']),
            total=len(all_data['ids'])
        )):
            reg_chunk = {
                'chunk_id': chunk_id,
                'content': text,
                'metadata': metadata
            }
            
            result = self.process_regulation_chunk(reg_chunk)
            results.append(result)
            
            self.stats['total_candidates'] += len(result['candidates'])
        
        self.stats['end_time'] = datetime.now()
        
        # Summary
        elapsed = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        _module_logger.info(f"Completed in {elapsed:.1f}s")
        _module_logger.info(f"Total candidates: {self.stats['total_candidates']}")
        _module_logger.info(f"Edge cases: {self.stats['edge_cases']}")
        
        return results
    
    def export_results(self, results: List[Dict], output_path: str):
        """Export results to JSON"""
        
        # FIX: Convert datetime to string - INDENTED INSIDE METHOD
        stats_fixed = {}
        for key, value in self.stats.items():
            if hasattr(value, 'isoformat'):  # Check if it's datetime
                stats_fixed[key] = value.isoformat()
            else:
                stats_fixed[key] = value
        
        export_data = {
            'generated_at': datetime.now().isoformat(),
            'pipeline_stats': stats_fixed,  # Use fixed version
            'logger_stats': self.logger.get_statistics(),
            'results': []
        }
        
        for r in results:
            export_data['results'].append({
                'regulation_chunk_id': r['regulation_chunk_id'],
                'candidates': [c.to_dict() for c in r['candidates']],
                'edge_case_classification': r['edge_case'].classification,
                'edge_case_reason': r['edge_case'].reason,
                'processing_time_ms': r['processing_time_ms']
            })
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        _module_logger.info(f"Exported to {output_path}")
    
    def get_summary(self) -> Dict:
        """Get pipeline summary"""
        return {
            'stats': self.stats,
            'logging': self.logger.get_statistics()
        }


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Day 21: Candidate Generation Pipeline - FULL INTEGRATION")
    print("=" * 60)
    
    # Initialize pipeline
    print("\n1. Initializing pipeline...")
    pipeline = CandidateGenerationPipeline(
        collection_name="regulations",
        top_k=3,
        min_score=0.3
    )
    
    # Process small sample
    print("\n2. Processing first 20 regulation chunks...")
    results = pipeline.process_all(limit=20)
    
    # Summary
    print("\n3. Summary:")
    summary = pipeline.get_summary()
    print(f"   Regulations processed: {summary['stats']['total_regulations']}")
    print(f"   Total candidates: {summary['stats']['total_candidates']}")
    print(f"   Edge cases: {summary['stats']['edge_cases']}")
    
    # Export
    print("\n4. Exporting results...")
    pipeline.export_results(results, "outputs/day21_candidate_results.json")
    
    # Show sample
    print("\n5. Sample result:")
    if results:
        r = results[0]
        print(f"   Regulation: {r['regulation_chunk_id']}")
        print(f"   Candidates: {len(r['candidates'])}")
        print(f"   Classification: {r['edge_case'].classification}")
        print(f"   Time: {r['processing_time_ms']:.1f}ms")
    
    print("\n" + "=" * 60)
    print("=" * 60)
    print("\nDeliverable: CandidateGenerationPipeline producing List[CandidatePair]")