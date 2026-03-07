"""Confidence Calibrator - Advanced confidence scoring and calibration"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Tuple, Optional
import numpy as np
import logging
from dataclasses import dataclass
from datetime import datetime
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import json

from scoring.gap_classifier import GapClassification, GapClass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of confidence calibration"""
    original_confidence: float
    calibrated_confidence: float
    method: str
    reliability: str  # high/medium/low
    
    def to_dict(self) -> Dict:
        return {
            'original': round(self.original_confidence, 3),
            'calibrated': round(self.calibrated_confidence, 3),
            'method': self.method,
            'reliability': self.reliability
        }


class ConfidenceCalibrator:
    """
    Calibrates confidence scores to improve reliability
    
    Methods:
    - Platt Scaling (sigmoid calibration)
    - Isotonic Regression
    - Weight optimization
    """
    
    def __init__(self, method: str = "platt"):
        self.method = method
        self.platt_model = None
        self.isotonic_model = None
        self.calibration_stats = {
            'samples_used': 0,
            'brier_score_before': 0.0,
            'brier_score_after': 0.0
        }
        
        # Default weights (will be optimized if calibration data provided)
        self.optimal_weights = {
            'cross_encoder': 0.50,
            'bi_encoder': 0.25,
            'candidate_rank': 0.15,
            'score_variance': 0.10
        }
    
    def fit(
        self,
        confidences: List[float],
        accuracies: List[bool]
    ) -> Dict:
        """
        Fit calibration model on validation data
        
        Args:
            confidences: Raw confidence scores
            accuracies: True/False if classification was correct
        
        Returns:
            Calibration statistics
        """
        X = np.array(confidences).reshape(-1, 1)
        y = np.array([1 if a else 0 for a in accuracies])
        
        # Calculate Brier score before calibration
        brier_before = np.mean((X.flatten() - y) ** 2)
        
        if self.method == "platt":
            self.platt_model = LogisticRegression()
            self.platt_model.fit(X, y)
            calibrated = self.platt_model.predict_proba(X)[:, 1]
        elif self.method == "isotonic":
            self.isotonic_model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
            self.isotonic_model.fit(X.flatten(), y)
            calibrated = self.isotonic_model.predict(X.flatten())
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        
        brier_after = np.mean((calibrated - y) ** 2)
        
        self.calibration_stats = {
            'samples_used': len(confidences),
            'brier_score_before': float(brier_before),
            'brier_score_after': float(brier_after),
            'improvement': float(brier_before - brier_after)
        }
        
        logger.info(f"Calibration fitted: Brier score {brier_before:.3f} → {brier_after:.3f}")
        
        return self.calibration_stats
    
    def calibrate(self, confidence: float) -> CalibrationResult:
        """
        Calibrate a single confidence score
        
        Args:
            confidence: Raw confidence score
        
        Returns:
            CalibrationResult with calibrated score
        """
        if self.platt_model is None and self.isotonic_model is None:
            # No calibration fitted, return original
            return CalibrationResult(
                original_confidence=confidence,
                calibrated_confidence=confidence,
                method="none",
                reliability="unknown"
            )
        
        X = np.array([[confidence]])
        
        if self.method == "platt" and self.platt_model:
            calibrated = self.platt_model.predict_proba(X)[0, 1]
        elif self.method == "isotonic" and self.isotonic_model:
            calibrated = self.isotonic_model.predict([confidence])[0]
        else:
            calibrated = confidence
        
        # Determine reliability based on calibration quality
        if self.calibration_stats.get('improvement', 0) > 0.05:
            reliability = "high"
        elif self.calibration_stats.get('improvement', 0) > 0.01:
            reliability = "medium"
        else:
            reliability = "low"
        
        return CalibrationResult(
            original_confidence=confidence,
            calibrated_confidence=float(calibrated),
            method=self.method,
            reliability=reliability
        )
    
    def optimize_weights(
        self,
        training_data: List[Dict]
    ) -> Dict:
        """
        Optimize confidence weights using grid search
        
        Args:
            training_data: List of dicts with features and correctness
        
        Returns:
            Optimal weights
        """
        logger.info("Optimizing confidence weights...")
        
        best_weights = None
        best_score = float('inf')
        
        # Grid search over weight combinations
        # w_ce + w_bi + w_rank + w_var = 1.0
        grid_points = 5
        weights_range = np.linspace(0.1, 0.6, grid_points)
        
        for w_ce in weights_range:
            for w_bi in weights_range:
                for w_rank in weights_range:
                    w_var = 1.0 - w_ce - w_bi - w_rank
                    if w_var < 0.05 or w_var > 0.4:
                        continue
                    
                    score = self._evaluate_weights(
                        {'cross_encoder': w_ce, 'bi_encoder': w_bi, 
                         'candidate_rank': w_rank, 'score_variance': w_var},
                        training_data
                    )
                    
                    if score < best_score:
                        best_score = score
                        best_weights = {
                            'cross_encoder': round(w_ce, 2),
                            'bi_encoder': round(w_bi, 2),
                            'candidate_rank': round(w_rank, 2),
                            'score_variance': round(w_var, 2)
                        }
        
        self.optimal_weights = best_weights
        logger.info(f"Optimal weights found: {best_weights}")
        
        return best_weights
    
    def _evaluate_weights(self, weights: Dict, data: List[Dict]) -> float:
        """Evaluate weight configuration (lower is better)"""
        errors = []
        
        for item in data:
            # Calculate confidence with these weights
            conf = (
                weights['cross_encoder'] * item.get('cross_encoder_score', 0) +
                weights['bi_encoder'] * item.get('bi_encoder_score', 0) +
                weights['candidate_rank'] * item.get('rank_score', 0) +
                weights['score_variance'] * item.get('variance_score', 0)
            )
            
            # Error is difference from actual correctness (0 or 1)
            actual = 1 if item.get('is_correct', False) else 0
            errors.append((conf - actual) ** 2)
        
        return np.mean(errors)
    
    def calculate_confidence_interval(
        self,
        confidence: float,
        sample_size: int = 1
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval using Wilson score
        
        Args:
            confidence: Point estimate
            sample_size: Number of samples (affects interval width)
        
        Returns:
            (lower_bound, upper_bound)
        """
        # Wilson score interval
        z = 1.96  # 95% confidence
        n = max(sample_size, 1)
        p = confidence
        
        denominator = 1 + z**2 / n
        centre_adjusted_probability = p + z*z / (2*n)
        adjusted_standard_deviation = np.sqrt((p*(1-p) + z*z / (4*n)) / n)
        
        lower_bound = (centre_adjusted_probability - z*adjusted_standard_deviation) / denominator
        upper_bound = (centre_adjusted_probability + z*adjusted_standard_deviation) / denominator
        
        return max(0.0, lower_bound), min(1.0, upper_bound)
    
    def get_confidence_level_with_interval(
        self,
        confidence: float,
        sample_size: int = 1
    ) -> Dict:
        """Get confidence level with uncertainty interval"""
        lower, upper = self.calculate_confidence_interval(confidence, sample_size)
        
        # Determine level based on calibrated confidence
        if confidence >= 0.8:
            level = 'high'
        elif confidence >= 0.6:
            level = 'medium'
        else:
            level = 'low'
        
        return {
            'point_estimate': round(confidence, 3),
            'confidence_level': level,
            'interval_95': [round(lower, 3), round(upper, 3)],
            'interval_width': round(upper - lower, 3),
            'reliability': 'high' if (upper - lower) < 0.2 else 'medium'
        }


class ConfidenceValidator:
    """Validates confidence reliability on test sets"""
    
    def __init__(self):
        self.validation_results = []
    
    def test_reliability(
        self,
        classifications: List[GapClassification],
        ground_truth: Dict[str, bool]  # chunk_id -> is_correct
    ) -> Dict:
        """
        Test if confidence scores correlate with accuracy
        
        Args:
            classifications: List of classifications with confidence
            ground_truth: Dict mapping chunk_id to correctness
        
        Returns:
            Reliability metrics
        """
        results = []
        
        for cls in classifications:
            is_correct = ground_truth.get(cls.regulation_chunk_id, False)
            results.append({
                'confidence': cls.confidence,
                'predicted_correct': cls.confidence > 0.5,
                'actual_correct': is_correct
            })
        
        # Bin by confidence levels
        bins = {'high': [], 'medium': [], 'low': []}
        for r in results:
            if r['confidence'] >= 0.8:
                bins['high'].append(r['actual_correct'])
            elif r['confidence'] >= 0.6:
                bins['medium'].append(r['actual_correct'])
            else:
                bins['low'].append(r['actual_correct'])
        
        reliability = {}
        for level, correct_list in bins.items():
            if correct_list:
                accuracy = sum(correct_list) / len(correct_list)
                reliability[level] = {
                    'count': len(correct_list),
                    'actual_accuracy': round(accuracy, 3),
                    'expected_min_confidence': 0.8 if level == 'high' else (0.6 if level == 'medium' else 0.0),
                    'is_reliable': accuracy >= (0.8 if level == 'high' else (0.6 if level == 'medium' else 0.3))
                }
        
        # Overall correlation
        confidences = [r['confidence'] for r in results]
        accuracies = [1 if r['actual_correct'] else 0 for r in results]
        
        if len(set(confidences)) > 1:
            correlation = np.corrcoef(confidences, accuracies)[0, 1]
        else:
            correlation = 0.0
        
        return {
            'reliability_by_level': reliability,
            'confidence_accuracy_correlation': round(float(correlation), 3),
            'total_tested': len(results),
            'overall_accuracy': round(sum(accuracies) / len(accuracies), 3) if accuracies else 0
        }


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Day 27: Confidence Calibrator Test")
    print("=" * 60)
    
    # Test calibration fitting
    print("\n1. Testing calibration fitting...")
    
    # Generate synthetic calibration data
    np.random.seed(42)
    n_samples = 100
    
    # Simulate confidences and accuracies (with some noise)
    raw_confidences = np.random.beta(2, 2, n_samples)
    # Make it imperfectly calibrated
    actual_accuracies = [c > np.random.uniform(0.3, 0.7) for c in raw_confidences]
    
    calibrator = ConfidenceCalibrator(method="isotonic")
    stats = calibrator.fit(raw_confidences.tolist(), actual_accuracies)
    
    print(f"   Samples: {stats['samples_used']}")
    print(f"   Brier score: {stats['brier_score_before']:.3f} → {stats['brier_score_after']:.3f}")
    print(f"   Improvement: {stats['improvement']:.3f}")
    
    # Test calibration
    print("\n2. Testing individual calibration...")
    test_confs = [0.3, 0.5, 0.7, 0.9]
    for conf in test_confs:
        result = calibrator.calibrate(conf)
        print(f"   {conf:.1f} → {result.calibrated_confidence:.3f} "
              f"({result.method}, {result.reliability})")
    
    # Test weight optimization
    print("\n3. Testing weight optimization...")
    training_data = [
        {
            'cross_encoder_score': np.random.uniform(0.5, 1.0),
            'bi_encoder_score': np.random.uniform(0.3, 0.9),
            'rank_score': 1.0 / np.random.randint(1, 5),
            'variance_score': np.random.uniform(0, 0.5),
            'is_correct': np.random.choice([True, False])
        }
        for _ in range(50)
    ]
    
    optimal = calibrator.optimize_weights(training_data)
    print(f"   Optimal weights: {optimal}")
    
    # Test confidence intervals
    print("\n4. Testing confidence intervals...")
    for conf in [0.9, 0.7, 0.5]:
        interval = calibrator.calculate_confidence_interval(conf, sample_size=10)
        info = calibrator.get_confidence_level_with_interval(conf, sample_size=10)
        print(f"   Confidence {conf}: interval [{interval[0]:.2f}, {interval[1]:.2f}], "
              f"width {info['interval_width']:.3f}")
    
    # Test reliability validation
    print("\n5. Testing reliability validation...")
    validator = ConfidenceValidator()
    
    # Create mock classifications
    mock_classifications = []
    ground_truth = {}
    
    for i in range(30):
        # Create correlation between confidence and correctness
        is_correct = np.random.random() > 0.3
        confidence = 0.9 if is_correct else np.random.uniform(0.4, 0.7)
        
        cls = GapClassification(
            regulation_chunk_id=f"reg_{i}",
            regulation_text="text",
            regulation_metadata={},
            classification=GapClass.ALIGNED if is_correct else GapClass.PARTIAL,
            confidence=confidence,
            confidence_level='high' if confidence > 0.8 else 'medium',
            bi_encoder_score=0.6,
            cross_encoder_score=confidence,
            final_score=confidence,
            threshold_min=0.0,
            threshold_max=1.0,
            reasoning="test",
            recommended_action="none",
            policy_matches=[],
            classified_at=datetime.now().isoformat(),
            config_version="test"
        )
        mock_classifications.append(cls)
        ground_truth[f"reg_{i}"] = is_correct
    
    reliability = validator.test_reliability(mock_classifications, ground_truth)
    print(f"   Correlation: {reliability['confidence_accuracy_correlation']:.3f}")
    print(f"   Overall accuracy: {reliability['overall_accuracy']:.3f}")
    print(f"   By level: {reliability['reliability_by_level']}")
    
    print("\n" + "=" * 60)
    print("=" * 60)