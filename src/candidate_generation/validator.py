"""Validator - Manual annotation and accuracy validation"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Optional
import json
import logging

from candidate_generation.candidate_generator import CandidatePair

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CandidateValidator:
    def __init__(self):
        self.annotations = {}
        self.metrics = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'precision_at_3': 0.0
        }
    
    def create_annotation_template(
        self,
        pairs: List[CandidatePair],
        output_path: str
    ):
        """Create template for manual annotation"""
        template = []
        
        for pair in pairs:
            template.append({
                'regulation_chunk_id': pair.regulation_chunk_id,
                'policy_chunk_id': pair.policy_chunk_id,
                'bi_encoder_score': pair.bi_encoder_score,
                'regulation_text': pair.regulation_text[:200],
                'policy_text': pair.policy_text[:200],
                'is_correct_match': None,  # To be filled manually
                'notes': ''
            })
        
        with open(output_path, 'w') as f:
            json.dump(template, f, indent=2)
        
        logger.info(f"Created annotation template: {output_path}")
        return template
    
    def load_annotations(self, annotation_path: str):
        """Load manual annotations"""
        with open(annotation_path) as f:
            annotated = json.load(f)
        
        for item in annotated:
            key = (item['regulation_chunk_id'], item['policy_chunk_id'])
            self.annotations[key] = item.get('is_correct_match', False)
        
        logger.info(f"Loaded {len(self.annotations)} annotations")
    
    def compute_metrics(
        self,
        candidates: List[CandidatePair],
        top_k: int = 3
    ) -> Dict:
        """
        Compute precision@k and other metrics
        
        Requires annotations to be loaded first
        """
        if not self.annotations:
            logger.warning("No annotations loaded. Run load_annotations() first.")
            return {}
        
        # Group by regulation
        by_regulation = {}
        for c in candidates:
            reg_id = c.regulation_chunk_id
            if reg_id not in by_regulation:
                by_regulation[reg_id] = []
            by_regulation[reg_id].append(c)
        
        # Compute precision@k for each regulation
        precisions = []
        
        for reg_id, cands in by_regulation.items():
            # Sort by score
            cands_sorted = sorted(cands, key=lambda x: x.bi_encoder_score, reverse=True)
            top_candidates = cands_sorted[:top_k]
            
            # Check which are correct
            correct_in_top = 0
            for c in top_candidates:
                key = (c.regulation_chunk_id, c.policy_chunk_id)
                if self.annotations.get(key, False):
                    correct_in_top += 1
            
            if top_candidates:
                precision = correct_in_top / len(top_candidates)
                precisions.append(precision)
        
        # Overall metrics
        avg_precision_at_k = sum(precisions) / len(precisions) if precisions else 0
        
        self.metrics['precision_at_3'] = avg_precision_at_k
        
        return {
            'precision_at_3': avg_precision_at_k,
            'num_regulations_evaluated': len(by_regulation),
            'num_annotations_used': len(self.annotations)
        }
    
    def identify_failure_patterns(
        self,
        candidates: List[CandidatePair]
    ) -> Dict:
        """Identify common failure patterns"""
        patterns = {
            'low_similarity_correct': [],  # Score < 0.5 but correct
            'high_similarity_incorrect': [],  # Score > 0.7 but wrong
            'missing_candidates': []  # Should have found but didn't
        }
        
        for c in candidates:
            key = (c.regulation_chunk_id, c.policy_chunk_id)
            is_correct = self.annotations.get(key, False)
            
            if is_correct and c.bi_encoder_score < 0.5:
                patterns['low_similarity_correct'].append({
                    'pair': key,
                    'score': c.bi_encoder_score
                })
            elif not is_correct and c.bi_encoder_score > 0.7:
                patterns['high_similarity_incorrect'].append({
                    'pair': key,
                    'score': c.bi_encoder_score
                })
        
        return patterns


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Day 20: Validator Test")
    print("=" * 60)
    
    validator = CandidateValidator()
    
    # Create sample candidates
    print("\n1. Creating sample candidates...")
    sample_candidates = [
        CandidatePair("reg_1", "Obligations for reporting", {}, "pol_1", "Reporting requirements", {}, 0.85, 1),
        CandidatePair("reg_1", "Obligations for reporting", {}, "pol_2", "Data protection rules", {}, 0.65, 2),
        CandidatePair("reg_2", "Customer identification", {}, "pol_3", "KYC procedures", {}, 0.75, 1),
    ]
    
    # Create annotation template
    print("\n2. Creating annotation template...")
    validator.create_annotation_template(sample_candidates, "outputs/day20_annotations_template.json")
    
    # Simulate loading annotations (normally you'd edit the file manually)
    print("\n3. Simulating annotations...")
    simulated_annotations = {
        ('reg_1', 'pol_1'): True,   # Correct match
        ('reg_1', 'pol_2'): False,  # Incorrect
        ('reg_2', 'pol_3'): True,   # Correct
    }
    validator.annotations = simulated_annotations
    
    # Compute metrics
    print("\n4. Computing metrics...")
    metrics = validator.compute_metrics(sample_candidates, top_k=2)
    print(f"   {metrics}")
    
    # Identify patterns
    print("\n5. Failure patterns...")
    patterns = validator.identify_failure_patterns(sample_candidates)
    print(f"   Low similarity but correct: {len(patterns['low_similarity_correct'])}")
    print(f"   High similarity but wrong: {len(patterns['high_similarity_incorrect'])}")
    
    print("\n" + "=" * 60)
    print("=" * 60)