"""Candidate Ranker - Multi-factor ranking algorithm"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict
from dataclasses import dataclass
import numpy as np
import logging

from candidate_generation.candidate_generator import CandidatePair

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RankedCandidate(CandidatePair):
    """Extended candidate with ranking scores"""
    section_match_score: float = 0.0
    final_score: float = 0.0
    
    def to_dict(self) -> Dict:
        base = super().to_dict()
        base.update({
            'section_match_score': self.section_match_score,
            'final_score': self.final_score
        })
        return base


class CandidateRanker:
    def __init__(
        self,
        similarity_weight: float = 0.7,
        section_weight: float = 0.3
    ):
        self.similarity_weight = similarity_weight
        self.section_weight = section_weight
        
        # Section type mappings
        self.section_mappings = {
            'obligation': ['requirement', 'mandatory', 'shall', 'must'],
            'prohibition': ['prohibited', 'forbidden', 'shall not', 'must not'],
            'permission': ['may', 'can', 'permitted', 'allowed'],
            'procedure': ['process', 'procedure', 'steps', 'how to'],
            'reporting': ['report', 'notify', 'inform', 'disclose'],
            'record': ['record', 'document', 'maintain', 'keep']
        }
    
    def extract_section_type(self, text: str) -> str:
        """Extract section type from text"""
        text_lower = text.lower()
        
        for section_type, keywords in self.section_mappings.items():
            if any(kw in text_lower for kw in keywords):
                return section_type
        
        return 'general'
    
    def calculate_section_match(
        self,
        reg_text: str,
        policy_text: str
    ) -> float:
        """
        Calculate section type match score
        
        Returns:
            1.0 if same section type, 0.5 if related, 0.0 if different
        """
        reg_type = self.extract_section_type(reg_text)
        policy_type = self.extract_section_type(policy_text)
        
        if reg_type == policy_type:
            return 1.0
        
        # Related types
        related = {
            'obligation': ['procedure', 'reporting', 'record'],
            'reporting': ['obligation', 'record', 'procedure'],
            'procedure': ['obligation', 'permission']
        }
        
        if policy_type in related.get(reg_type, []):
            return 0.5
        
        return 0.0
    
    def rank_candidates(
        self,
        candidates: List[CandidatePair]
    ) -> List[RankedCandidate]:
        """
        Apply multi-factor ranking to candidates
        
        Args:
            candidates: List of CandidatePair from bi-encoder
        
        Returns:
            List of RankedCandidate with final scores
        """
        ranked = []
        
        for candidate in candidates:
            # Section type matching
            section_score = self.calculate_section_match(
                candidate.regulation_text,
                candidate.policy_text
            )
            
            # Combined score
            final_score = (
                self.similarity_weight * candidate.bi_encoder_score +
                self.section_weight * section_score
            )
            
            ranked_cand = RankedCandidate(
                regulation_chunk_id=candidate.regulation_chunk_id,
                regulation_text=candidate.regulation_text,
                regulation_metadata=candidate.regulation_metadata,
                policy_chunk_id=candidate.policy_chunk_id,
                policy_text=candidate.policy_text,
                policy_metadata=candidate.policy_metadata,
                bi_encoder_score=candidate.bi_encoder_score,
                rank=candidate.rank,
                section_match_score=section_score,
                final_score=final_score
            )
            ranked.append(ranked_cand)
        
        # Sort by final score
        ranked.sort(key=lambda x: x.final_score, reverse=True)
        
        # Re-rank
        for i, cand in enumerate(ranked, 1):
            cand.rank = i
        
        return ranked
    
    def deduplicate_candidates(
        self,
        candidates: List[RankedCandidate],
        similarity_threshold: float = 0.95
    ) -> List[RankedCandidate]:
        """
        Remove highly similar policy chunks
        
        Args:
            candidates: Ranked candidates
            similarity_threshold: Merge if text similarity > threshold
        
        Returns:
            Deduplicated list
        """
        if not candidates:
            return candidates
        
        unique = [candidates[0]]
        
        for cand in candidates[1:]:
            # Simple text overlap check
            is_duplicate = False
            for existing in unique:
                # Check policy text similarity (simple Jaccard)
                set1 = set(cand.policy_text.lower().split())
                set2 = set(existing.policy_text.lower().split())
                
                if len(set1) > 0 and len(set2) > 0:
                    jaccard = len(set1 & set2) / len(set1 | set2)
                    if jaccard > similarity_threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique.append(cand)
        
        logger.info(f"Deduplicated: {len(candidates)} → {len(unique)}")
        return unique


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Day 17: Candidate Ranker Test")
    print("=" * 60)
    
    from candidate_generation.candidate_generator import CandidateGenerator
    
    # Generate some candidates
    print("\n1. Generating candidates...")
    gen = CandidateGenerator()
    chroma = gen.chroma
    
    coll = chroma.get_collection("regulations")
    sample = coll.get(limit=5)
    
    candidates = []
    for chunk_id, text, meta in zip(sample['ids'], sample['documents'], sample['metadatas']):
        reg_chunk = {
            'chunk_id': chunk_id,
            'content': text,
            'metadata': meta
        }
        cands = gen.get_candidates(reg_chunk, top_k=3, min_score=0.2)
        candidates.extend(cands)
    
    print(f"   Generated {len(candidates)} raw candidates")
    
    # Rank them
    print("\n2. Ranking candidates...")
    ranker = CandidateRanker(similarity_weight=0.7, section_weight=0.3)
    ranked = ranker.rank_candidates(candidates)
    
    print(f"   Top 3 after ranking:")
    for i, c in enumerate(ranked[:3], 1):
        print(f"   {i}. Final: {c.final_score:.3f} | "
              f"Similarity: {c.bi_encoder_score:.3f} | "
              f"Section: {c.section_match_score:.1f}")
        print(f"      Reg: {c.regulation_text[:50]}...")
        print(f"      Policy: {c.policy_text[:50]}...")
    
    # Deduplicate
    print("\n3. Deduplicating...")
    deduped = ranker.deduplicate_candidates(ranked, similarity_threshold=0.9)
    print(f"   After dedup: {len(deduped)}")
    
    print("\n" + "=" * 60)
    print("=" * 60)