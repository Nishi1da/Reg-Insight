"""Edge Case Handler - Handle no-match, ambiguous, and fallback scenarios"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np
import logging

from candidate_generation.candidate_generator import CandidatePair

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EdgeCaseResult:
    """Result with edge case classification"""
    regulation_chunk_id: str
    candidates: List[CandidatePair]
    classification: str  # 'no_match', 'ambiguous', 'clear_match', 'low_confidence'
    reason: str
    fallback_used: bool
    recommended_action: str


class EdgeCaseHandler:
    def __init__(
        self,
        no_match_threshold: float = 0.3,
        ambiguous_threshold: float = 0.05
    ):
        self.no_match_threshold = no_match_threshold
        self.ambiguous_threshold = ambiguous_threshold
    
    def classify_case(
        self,
        regulation_chunk_id: str,
        candidates: List[CandidatePair]
    ) -> EdgeCaseResult:
        """
        Classify the candidate retrieval case
        
        Returns:
            EdgeCaseResult with classification and recommendations
        """
        # No candidates found
        if not candidates:
            return EdgeCaseResult(
                regulation_chunk_id=regulation_chunk_id,
                candidates=[],
                classification='no_match',
                reason='No candidates above minimum similarity threshold',
                fallback_used=False,
                recommended_action='Flag for manual review - potential gap'
            )
        
        # Check if all scores are low
        max_score = max(c.bi_encoder_score for c in candidates)
        if max_score < self.no_match_threshold:
            return EdgeCaseResult(
                regulation_chunk_id=regulation_chunk_id,
                candidates=candidates,
                classification='no_match',
                reason=f'Best candidate score ({max_score:.3f}) below threshold ({self.no_match_threshold})',
                fallback_used=False,
                recommended_action='Potential gap - no adequate policy coverage'
            )
        
        # Check for ambiguity (top candidates very close)
        if len(candidates) >= 2:
            score_diff = candidates[0].bi_encoder_score - candidates[1].bi_encoder_score
            if score_diff < self.ambiguous_threshold:
                return EdgeCaseResult(
                    regulation_chunk_id=regulation_chunk_id,
                    candidates=candidates,
                    classification='ambiguous',
                    reason=f'Top 2 candidates have similar scores (diff: {score_diff:.3f})',
                    fallback_used=False,
                    recommended_action='Manual review required - unclear best match'
                )
        
        # Check for low confidence (borderline score)
        if max_score < 0.5:
            return EdgeCaseResult(
                regulation_chunk_id=regulation_chunk_id,
                candidates=candidates,
                classification='low_confidence',
                reason=f'Best score ({max_score:.3f}) is borderline',
                fallback_used=False,
                recommended_action='Review recommended - weak match'
            )
        
        # Clear match
        return EdgeCaseResult(
            regulation_chunk_id=regulation_chunk_id,
            candidates=candidates,
            classification='clear_match',
            reason=f'Clear best candidate with score {max_score:.3f}',
            fallback_used=False,
            recommended_action='Proceed with cross-encoder scoring'
        )
    
    def apply_fallback(
        self,
        regulation_chunk: Dict,
        chroma_manager,
        collection_name: str = "regulations"
    ) -> List[CandidatePair]:
        """
        Apply fallback strategies when semantic search fails
        
        Strategies:
        1. Keyword search using section headers
        2. Broader similarity threshold
        3. Exact phrase matching
        """
        logger.info(f"Applying fallback for {regulation_chunk['chunk_id']}")
        
        fallback_candidates = []
        
        # Strategy 1: Section header exact match
        section_header = regulation_chunk['metadata'].get('section_header', '')
        if section_header:
            # This would need keyword search implementation
            logger.info(f"  Trying section header match: {section_header[:50]}...")
            # Placeholder - would query with keywords
        
        # Strategy 2: Lower threshold
        # Re-query with lower threshold would happen here
        
        # Strategy 3: Expand to all collections
        
        return fallback_candidates
    
    def generate_edge_case_report(
        self,
        results: List[EdgeCaseResult]
    ) -> Dict:
        """Generate summary report of edge cases"""
        classifications = {}
        for r in results:
            classifications[r.classification] = classifications.get(r.classification, 0) + 1
        
        return {
            'total_processed': len(results),
            'no_match': classifications.get('no_match', 0),
            'ambiguous': classifications.get('ambiguous', 0),
            'low_confidence': classifications.get('low_confidence', 0),
            'clear_match': classifications.get('clear_match', 0),
            'fallback_used': sum(1 for r in results if r.fallback_used),
            'flagged_for_review': sum(1 for r in results if r.classification != 'clear_match')
        }


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Day 18: Edge Case Handler Test")
    print("=" * 60)
    
    handler = EdgeCaseHandler(no_match_threshold=0.3, ambiguous_threshold=0.05)
    
    # Test cases
    print("\n1. Testing classifications...")
    
    # Case 1: No candidates
    result1 = handler.classify_case("reg_1", [])
    print(f"   No candidates: {result1.classification} - {result1.reason[:50]}...")
    
    # Case 2: Low scores
    low_candidates = [
        CandidatePair("reg_1", "text", {}, "pol_1", "text", {}, 0.2, 1),
        CandidatePair("reg_1", "text", {}, "pol_2", "text", {}, 0.15, 2)
    ]
    result2 = handler.classify_case("reg_1", low_candidates)
    print(f"   Low scores: {result2.classification}")
    
    # Case 3: Ambiguous
    ambig_candidates = [
        CandidatePair("reg_1", "text", {}, "pol_1", "text", {}, 0.75, 1),
        CandidatePair("reg_1", "text", {}, "pol_2", "text", {}, 0.73, 2)
    ]
    result3 = handler.classify_case("reg_1", ambig_candidates)
    print(f"   Ambiguous (diff 0.02): {result3.classification}")
    
    # Case 4: Clear match
    clear_candidates = [
        CandidatePair("reg_1", "text", {}, "pol_1", "text", {}, 0.85, 1),
        CandidatePair("reg_1", "text", {}, "pol_2", "text", {}, 0.60, 2)
    ]
    result4 = handler.classify_case("reg_1", clear_candidates)
    print(f"   Clear match: {result4.classification}")
    
    # Generate report
    print("\n2. Generating report...")
    all_results = [result1, result2, result3, result4]
    report = handler.generate_edge_case_report(all_results)
    print(f"   {report}")
    
    print("\n" + "=" * 60)
    print("=" * 60)