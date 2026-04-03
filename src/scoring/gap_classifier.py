"""Gap Classifier - Regulation to Policy Gap Classification"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import yaml
import json
import numpy as np
import logging
from datetime import datetime

from scoring.precision_pipeline import PrecisionScoredCandidate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GapClass(Enum):
    """Gap classification categories"""
    ALIGNED = "aligned"
    PARTIAL = "partial"
    GAP = "gap"
    UNMATCHED = "unmatched"  # No candidates found


@dataclass
class GapClassification:
    """Complete gap classification result"""
    regulation_chunk_id: str
    regulation_text: str
    regulation_metadata: Dict
    
    classification: GapClass
    confidence: float  # 0-1
    confidence_level: str  # high/medium/low
    
    # Scores
    bi_encoder_score: float
    cross_encoder_score: float
    final_score: float
    
    # Threshold info
    threshold_min: float
    threshold_max: float
    
    # Reasoning
    reasoning: str
    recommended_action: str
    
    # All matches (for context)
    policy_matches: List[Dict]
    
    # Metadata
    classified_at: str
    config_version: str
    
    def to_dict(self) -> Dict:
        return {
            'regulation_chunk_id': self.regulation_chunk_id,
            'regulation_text': self.regulation_text[:200],
            'classification': self.classification.value,
            'confidence': round(self.confidence, 3),
            'confidence_level': self.confidence_level,
            'scores': {
                'bi_encoder': round(self.bi_encoder_score, 3),
                'cross_encoder': round(self.cross_encoder_score, 3),
                'final': round(self.final_score, 3)
            },
            'threshold_range': [self.threshold_min, self.threshold_max],
            'reasoning': self.reasoning,
            'recommended_action': self.recommended_action,
            'num_policy_matches': len(self.policy_matches),
            'classified_at': self.classified_at,
            'config_version': self.config_version
        }


class GapClassifier:
    """
    Classifies regulation-policy matches into gap categories
    with confidence scoring
    """
    
    def __init__(self, config_path: str = "config/classification_config.yaml"):
        self.config = self._load_config(config_path)
        self.thresholds = self.config['thresholds']
        self.confidence_weights = self.config['confidence_weights']
        self.scoring_config = self.config['scoring']
        self.config_version = datetime.now().strftime("%Y%m%d")
        
        self.stats = {
            'total_classified': 0,
            'by_category': {cat: 0 for cat in ['aligned', 'partial', 'gap', 'unmatched']},
            'avg_confidence': 0.0
        }
    
    def _load_config(self, path: str) -> Dict:
        """Load classification configuration"""
        default_config = {
            'thresholds': {
                'aligned': {'min_score': 0.7, 'max_score': 1.0, 'description': 'Aligned', 'action': 'None'},
                'partial': {'min_score': 0.4, 'max_score': 0.69, 'description': 'Partial', 'action': 'Review'},
                'gap': {'min_score': 0.0, 'max_score': 0.39, 'description': 'Gap', 'action': 'Update'}
            },
            'confidence_weights': {
                'cross_encoder': 0.5, 'bi_encoder': 0.25, 'candidate_rank': 0.15, 'score_variance': 0.1
            },
            'scoring': {'bi_encoder_weight': 0.3, 'cross_encoder_weight': 0.7}
        }
        
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config not found at {path}, using defaults")
            return default_config
    
    def classify(
        self,
        regulation_chunk_id: str,
        regulation_text: str,
        regulation_metadata: Dict,
        scored_candidates: List[PrecisionScoredCandidate]
    ) -> GapClassification:
        """
        Classify regulation chunk based on scored candidates
        
        Args:
            regulation_chunk_id: ID of regulation chunk
            regulation_text: Regulation text
            regulation_metadata: Regulation metadata
            scored_candidates: List of precision-scored candidates (can be empty)
        
        Returns:
            GapClassification with category and confidence
        """
        self.stats['total_classified'] += 1
        
        # Handle unmatched case
        if not scored_candidates:
            self.stats['by_category']['unmatched'] += 1
            
            return GapClassification(
                regulation_chunk_id=regulation_chunk_id,
                regulation_text=regulation_text,
                regulation_metadata=regulation_metadata,
                classification=GapClass.UNMATCHED,
                confidence=0.0,
                confidence_level='low',
                bi_encoder_score=0.0,
                cross_encoder_score=0.0,
                final_score=0.0,
                threshold_min=0.0,
                threshold_max=0.0,
                reasoning="No policy candidates found above minimum threshold",
                recommended_action="Create new policy - no coverage exists",
                policy_matches=[],
                classified_at=datetime.now().isoformat(),
                config_version=self.config_version
            )
        
        # Get best match
        best_candidate = scored_candidates[0]
        final_score = best_candidate.final_score

        
        MIN_FINAL_SCORE = 0.15

        if final_score < MIN_FINAL_SCORE:
            self.stats['by_category']['unmatched'] += 1
            
            return GapClassification(
                regulation_chunk_id=regulation_chunk_id,
                regulation_text=regulation_text,
                regulation_metadata=regulation_metadata,
                classification=GapClass.UNMATCHED,
                confidence=0.85,
                confidence_level='high',
                bi_encoder_score=best_candidate.bi_encoder_score,
                cross_encoder_score=best_candidate.cross_encoder_score,
                final_score=final_score,
                threshold_min=0.0,
                threshold_max=MIN_FINAL_SCORE,
                reasoning=f"Best policy match {final_score:.3f} is below unmatched threshold — no relevant policy exists",
                recommended_action="No existing policy covers this requirement — create new policy",
                policy_matches=[],
                classified_at=datetime.now().isoformat(),
                config_version=self.config_version
    )
        
        # Determine classification
        if final_score >= self.thresholds['aligned']['min_score']:
            classification = GapClass.ALIGNED
            threshold_info = self.thresholds['aligned']
        elif final_score >= self.thresholds['partial']['min_score']:
            classification = GapClass.PARTIAL
            threshold_info = self.thresholds['partial']
        else:
            classification = GapClass.GAP
            threshold_info = self.thresholds['gap']
            
            self.stats['by_category'][classification.value] += 1
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            best_candidate, scored_candidates, classification
        )
        
        confidence_level = self._get_confidence_level(confidence)
        
        # Build policy matches list
        policy_matches = [c.to_dict() for c in scored_candidates[:3]]
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            classification, best_candidate, scored_candidates, confidence
        )
        
        return GapClassification(
            regulation_chunk_id=regulation_chunk_id,
            regulation_text=regulation_text,
            regulation_metadata=regulation_metadata,
            classification=classification,
            confidence=confidence,
            confidence_level=confidence_level,
            bi_encoder_score=best_candidate.bi_encoder_score,
            cross_encoder_score=best_candidate.cross_encoder_score,
            final_score=final_score,
            threshold_min=threshold_info['min_score'],
            threshold_max=threshold_info['max_score'],
            reasoning=reasoning,
            recommended_action=threshold_info['action'],
            policy_matches=policy_matches,
            classified_at=datetime.now().isoformat(),
            config_version=self.config_version
        )
    
    def _calculate_confidence(
        self,
        best_candidate: PrecisionScoredCandidate,
        all_candidates: List[PrecisionScoredCandidate],
        classification: GapClass
    ) -> float:
        """
        Calculate confidence score based on multiple signals
        
        Formula: w1*cross_encoder + w2*bi_encoder + w3*rank + w4*variance
        """
        w = self.confidence_weights
        
        # Cross-encoder confidence (higher weight on precise model)
        ce_confidence = best_candidate.cross_encoder_score * w['cross_encoder']
        
        # Bi-encoder confidence
        bi_confidence = best_candidate.bi_encoder_score * w['bi_encoder']
        
        # Rank confidence (top rank = higher confidence)
        rank_score = 1.0 / best_candidate.rank_final if best_candidate.rank_final > 0 else 0
        rank_confidence = rank_score * w['candidate_rank']
        
        # Score variance (gap between top 2)
        variance_confidence = 0.0
        if len(all_candidates) >= 2:
            score_gap = best_candidate.final_score - all_candidates[1].final_score
            # Normalize to 0-1 (gap of 0.2 is considered high confidence)
            variance_confidence = min(score_gap / 0.2, 1.0) * w['score_variance']
        
        total_confidence = ce_confidence + bi_confidence + rank_confidence + variance_confidence
        
        # Boost confidence for aligned, reduce for gap
        if classification == GapClass.ALIGNED:
            total_confidence = min(total_confidence * 1.1, 1.0)
        elif classification == GapClass.GAP:
            # For gaps, confidence is based on lack of good matches
            total_confidence = 1.0 - best_candidate.final_score
        
        return round(total_confidence, 3)
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Convert numeric confidence to level"""
        thresholds = self.scoring_config['confidence_levels']
        if confidence >= thresholds['high']:
            return 'high'
        elif confidence >= thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _generate_reasoning(
        self,
        classification: GapClass,
        best: PrecisionScoredCandidate,
        all_candidates: List[PrecisionScoredCandidate],
        confidence: float
    ) -> str:
        """Generate human-readable reasoning"""
        reasons = []
        
        if classification == GapClass.ALIGNED:
            reasons.append(f"Strong match found with score {best.final_score:.2f}")
            if best.cross_encoder_score > 0.8:
                reasons.append("Cross-encoder confirms high semantic similarity")
            if len(all_candidates) > 1 and (best.final_score - all_candidates[1].final_score) > 0.1:
                reasons.append("Clear distinction from other candidates")
        
        elif classification == GapClass.PARTIAL:
            reasons.append(f"Moderate match with score {best.final_score:.2f}")
            reasons.append("Policy covers some aspects but may be incomplete")
            if best.cross_encoder_score < best.bi_encoder_score:
                reasons.append("Cross-encoder indicates less precise match than bi-encoder suggested")
        
        elif classification == GapClass.GAP:
            reasons.append(f"Best match score {best.final_score:.2f} below threshold")
            if best.cross_encoder_score < 0.3:
                reasons.append("Cross-encoder indicates poor semantic alignment")
            reasons.append("Current policy does not adequately address this requirement")
        
        elif classification == GapClass.UNMATCHED:
            reasons.append("No candidate policies found")
            reasons.append("Requirement may be entirely missing from policy corpus")
        
        return "; ".join(reasons)
    
    def get_score_distribution(self, classifications: List[GapClassification]) -> Dict:
        """Analyze score distribution by classification"""
        scores_by_class = {cat.value: [] for cat in GapClass}
        
        for c in classifications:
            scores_by_class[c.classification.value].append(c.final_score)
        
        distribution = {}
        for class_name, scores in scores_by_class.items():
            if scores:
                distribution[class_name] = {
                    'count': len(scores),
                    'mean': round(np.mean(scores), 3),
                    'std': round(np.std(scores), 3),
                    'min': round(min(scores), 3),
                    'max': round(max(scores), 3),
                    'percentiles': {
                        '25': round(np.percentile(scores, 25), 3),
                        '50': round(np.percentile(scores, 50), 3),
                        '75': round(np.percentile(scores, 75), 3)
                    }
                }
        
        return distribution
    
    def visualize_distribution(self, classifications: List[GapClassification]) -> str:
        """Create ASCII visualization of score distribution"""
        scores_by_class = {cat.value: [] for cat in GapClass}
        for c in classifications:
            scores_by_class[c.classification.value].append(c.final_score)
        
        lines = ["\nScore Distribution by Classification", "=" * 50]
        
        for class_name in ['aligned', 'partial', 'gap', 'unmatched']:
            scores = scores_by_class.get(class_name, [])
            if not scores:
                continue
            
            count = len(scores)
            mean = np.mean(scores) if scores else 0
            
            # Simple histogram
            bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            hist = np.histogram(scores, bins=bins)[0]
            max_hist = max(hist) if max(hist) > 0 else 1
            
            lines.append(f"\n{class_name.upper()} (n={count}, mean={mean:.2f}):")
            for i, (bin_start, count_in_bin) in enumerate(zip(bins[:-1], hist)):
                bar = "█" * int(20 * count_in_bin / max_hist)
                lines.append(f"  {bin_start:.1f}-{bins[i+1]:.1f}: {bar} ({count_in_bin})")
        
        return "\n".join(lines)
    
    def get_stats(self) -> Dict:
        """Get classification statistics"""
        if self.stats['total_classified'] > 0:
            self.stats['avg_confidence'] = sum(
                self.stats['by_category'].values()
            ) / self.stats['total_classified']
        return self.stats.copy()


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Day 24: Gap Classifier Test")
    print("=" * 60)
    
    # Initialize classifier
    print("\n1. Initializing gap classifier...")
    classifier = GapClassifier()
    print(f"   Thresholds: Aligned ≥{classifier.thresholds['aligned']['min_score']}, "
          f"Partial {classifier.thresholds['partial']['min_score']}-{classifier.thresholds['partial']['max_score']}, "
          f"Gap <{classifier.thresholds['gap']['max_score']}")
    
    # Test classifications
    print("\n2. Testing different classifications...")
    
    from scoring.precision_pipeline import PrecisionScoredCandidate
    
    test_cases = [
        # (bi_score, cross_score, expected_class)
        (0.85, 0.90, "aligned"),
        (0.60, 0.65, "partial"),
        (0.30, 0.25, "gap"),
    ]
    
    for bi, cross, expected in test_cases:
        # Create mock candidate
        mock_candidates = [
            PrecisionScoredCandidate(
                regulation_chunk_id=f"reg_{expected}",
                regulation_text=f"Test regulation for {expected}",
                regulation_metadata={'source': 'test'},
                policy_chunk_id=f"pol_{expected}",
                policy_text=f"Test policy for {expected}",
                policy_metadata={'source': 'test'},
                bi_encoder_score=bi,
                cross_encoder_score=cross,
                final_score=0.3*bi + 0.7*cross,
                rank_bi=1,
                rank_cross=1,
                rank_final=1,
                inference_time_ms=50.0,
                scored_at=datetime.now().isoformat()
            )
        ]
        
        result = classifier.classify(
            f"reg_{expected}",
            f"Test regulation {expected}",
            {'source': 'test'},
            mock_candidates
        )
        
        print(f"   Scores (Bi: {bi:.2f}, Cross: {cross:.2f}) → "
              f"{result.classification.value.upper()} "
              f"(confidence: {result.confidence:.2f}, {result.confidence_level})")
        print(f"     Reasoning: {result.reasoning[:60]}...")
    
    # Test unmatched
    print("\n3. Testing unmatched case...")
    unmatched = classifier.classify("reg_unmatched", "Test", {}, [])
    print(f"   Classification: {unmatched.classification.value}")
    print(f"   Action: {unmatched.recommended_action}")
    
    # Test distribution analysis
    print("\n4. Testing distribution analysis...")
    all_classifications = []
    
    # Generate sample classifications
    for i in range(20):
        score = np.random.beta(2, 2)  # Random scores for demo
        mock = [PrecisionScoredCandidate(
            f"reg_{i}", "text", {}, f"pol_{i}", "text", {},
            score, score, score, 1, 1, 1, 50.0, datetime.now().isoformat()
        )]
        cls = classifier.classify(f"reg_{i}", "text", {}, mock)
        all_classifications.append(cls)
    
    dist = classifier.get_score_distribution(all_classifications)
    print(f"   Distribution by class:")
    for class_name, stats in dist.items():
        print(f"     {class_name}: {stats.get('count', 0)} items, mean={stats.get('mean', 0):.2f}")
    
    # Visualization
    print(classifier.visualize_distribution(all_classifications))
    
    # Stats
    print("\n5. Classification statistics:")
    stats = classifier.get_stats()
    print(f"   Total: {stats['total_classified']}")
    print(f"   By category: {stats['by_category']}")
    
    print("\n" + "=" * 60)
    print("=" * 60)