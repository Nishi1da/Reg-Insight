"""Unsupported Requirements Detector - Find regulations with no policy coverage"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Optional, Set, Tuple  # Add Tuple here
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import re

from candidate_generation.candidate_generator import CandidateGenerator
from scoring.gap_classifier import GapClassifier, GapClass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class UnsupportedRequirement:
    """Represents an unsupported regulation requirement"""
    regulation_chunk_id: str
    regulation_text: str
    regulation_source: str
    regulation_page: Optional[int]
    
    # Severity assessment
    severity: str  # critical, high, medium, low
    severity_score: float  # 0-1
    
    # Detection info
    detection_method: str  # no_candidates, low_scores, keyword_indicators
    confidence: float
    
    # Context
    section_header: Optional[str]
    requirement_type: str  # obligation, prohibition, permission, etc.
    
    # Recommendation
    recommended_priority: str
    estimated_effort: str  # high/medium/low
    
    def to_dict(self) -> Dict:
        return {
            'regulation_chunk_id': self.regulation_chunk_id,
            'regulation_text': self.regulation_text[:200],
            'source': self.regulation_source,
            'page': self.regulation_page,
            'severity': self.severity,
            'severity_score': round(self.severity_score, 2),
            'detection_method': self.detection_method,
            'confidence': round(self.confidence, 2),
            'requirement_type': self.requirement_type,
            'recommended_priority': self.recommended_priority,
            'estimated_effort': self.estimated_effort
        }


class UnsupportedRequirementsDetector:
    """
    Detects regulation requirements that lack policy support
    
    Two detection modes:
    1. No candidates: Semantic search returned nothing
    2. Low scores: All candidates below gap threshold
    """
    
    # Severity indicators
    CRITICAL_KEYWORDS = [
        'must', 'shall', 'required', 'mandatory', 'obligation',
        'comply', 'compliance', 'violation', 'penalty', 'fine',
        'sanction', 'prohibited', 'forbidden', 'illegal'
    ]
    
    HIGH_KEYWORDS = [
        'should', 'recommend', 'important', 'significant',
        'risk', 'security', 'protect', 'safeguard'
    ]
    
    PROCEDURE_KEYWORDS = [
        'process', 'procedure', 'implement', 'establish',
        'maintain', 'document', 'record', 'review'
    ]
    
    def __init__(
        self,
        severity_threshold_critical: int = 3,
        severity_threshold_high: int = 2
    ):
        self.severity_threshold_critical = severity_threshold_critical
        self.severity_threshold_high = severity_threshold_high
        self.detection_stats = {
            'total_checked': 0,
            'unsupported_found': 0,
            'by_severity': {'critical': 0, 'high': 0, 'medium': 0, 'low': 0},
            'by_method': {'no_candidates': 0, 'low_scores': 0, 'keyword_indicators': 0}
        }
    
    def analyze_requirement_type(self, text: str) -> str:
        """Determine requirement type from text"""
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in ['shall not', 'must not', 'prohibited', 'forbidden']):
            return 'prohibition'
        elif any(kw in text_lower for kw in ['shall', 'must', 'required', 'obligation']):
            return 'obligation'
        elif any(kw in text_lower for kw in ['may', 'can', 'permitted', 'allowed']):
            return 'permission'
        elif any(kw in text_lower for kw in ['should', 'recommend', 'advised']):
            return 'recommendation'
        else:
            return 'general'
    
    def calculate_severity(self, text: str, requirement_type: str) -> Tuple[str, float]:
        """
        Calculate severity based on keywords and requirement type
        
        Returns: (severity_level, severity_score)
        """
        text_lower = text.lower()
        
        # Count keyword matches
        critical_matches = sum(1 for kw in self.CRITICAL_KEYWORDS if kw in text_lower)
        high_matches = sum(1 for kw in self.HIGH_KEYWORDS if kw in text_lower)
        procedure_matches = sum(1 for kw in self.PROCEDURE_KEYWORDS if kw in text_lower)
        
        # Base score from keywords
        score = (critical_matches * 0.3) + (high_matches * 0.15) + (procedure_matches * 0.1)
        
        # Boost for obligation/prohibition types
        if requirement_type in ['obligation', 'prohibition']:
            score *= 1.3
        
        # Cap at 1.0
        score = min(score, 1.0)
        
        # Determine level
        if critical_matches >= self.severity_threshold_critical or score >= 0.7:
            severity = 'critical'
        elif critical_matches >= 1 or high_matches >= self.severity_threshold_high or score >= 0.5:
            severity = 'high'
        elif score >= 0.3:
            severity = 'medium'
        else:
            severity = 'low'
        
        return severity, round(score, 2)
    
    def detect_unsupported(
        self,
        regulation_chunk: Dict,
        candidates: Optional[List] = None,
        gap_classification: Optional[GapClass] = None
    ) -> Optional[UnsupportedRequirement]:
        """
        Detect if a regulation requirement is unsupported
        
        Args:
            regulation_chunk: Dict with chunk_id, content, metadata
            candidates: List of candidates (None if no candidates found)
            gap_classification: Pre-computed gap classification
        
        Returns:
            UnsupportedRequirement if unsupported, None otherwise
        """
        self.detection_stats['total_checked'] += 1
        
        text = regulation_chunk['content']
        chunk_id = regulation_chunk['chunk_id']
        metadata = regulation_chunk.get('metadata', {})
        
        # Determine detection method
        detection_method = None
        confidence = 0.0
        
        if candidates is None or len(candidates) == 0:
            detection_method = 'no_candidates'
            confidence = 0.9  # High confidence - semantic search found nothing
        elif gap_classification == GapClass.GAP:
            detection_method = 'low_scores'
            confidence = 0.7
        else:
            # Check if it's a clear gap despite having candidates
            if candidates and max(c.bi_encoder_score for c in candidates) < 0.3:
                detection_method = 'low_scores'
                confidence = 0.6
        
        if not detection_method:
            return None  # Supported requirement
        
        self.detection_stats['unsupported_found'] += 1
        self.detection_stats['by_method'][detection_method] += 1
        
        # Analyze requirement
        req_type = self.analyze_requirement_type(text)
        severity, severity_score = self.calculate_severity(text, req_type)
        
        self.detection_stats['by_severity'][severity] += 1
        
        # Determine priority and effort
        priority_map = {
            'critical': 'immediate',
            'high': 'high',
            'medium': 'medium',
            'low': 'low'
        }
        
        effort_map = {
            'obligation': 'high',
            'prohibition': 'medium',
            'permission': 'low',
            'recommendation': 'low',
            'general': 'medium'
        }
        
        return UnsupportedRequirement(
            regulation_chunk_id=chunk_id,
            regulation_text=text,
            regulation_source=metadata.get('source', 'unknown'),
            regulation_page=metadata.get('page_number'),
            severity=severity,
            severity_score=severity_score,
            detection_method=detection_method,
            confidence=confidence,
            section_header=metadata.get('section_header'),
            requirement_type=req_type,
            recommended_priority=priority_map.get(severity, 'medium'),
            estimated_effort=effort_map.get(req_type, 'medium')
        )
    
    def scan_collection(
        self,
        generator: CandidateGenerator,
        classifier: GapClassifier,
        limit: Optional[int] = None
    ) -> List[UnsupportedRequirement]:
        """
        Scan entire regulation collection for unsupported requirements
        
        Args:
            generator: CandidateGenerator instance
            classifier: GapClassifier instance
            limit: Optional limit for testing
        
        Returns:
            List of UnsupportedRequirement
        """
        logger.info("Scanning collection for unsupported requirements...")
        
        # Get all regulation chunks
        collection = generator.chroma.get_collection(generator.reg_collection)
        all_data = collection.get(limit=limit)
        
        unsupported = []
        
        for i, (chunk_id, text, metadata) in enumerate(zip(
            all_data['ids'], all_data['documents'], all_data['metadatas']
        )):
            if i % 50 == 0:
                logger.info(f"  Checked {i}/{len(all_data['ids'])} chunks...")
            
            reg_chunk = {
                'chunk_id': chunk_id,
                'content': text,
                'metadata': metadata
            }
            
            # Generate candidates
            candidates = generator.get_candidates(reg_chunk, top_k=3, min_score=0.2)
            
            # Classify (simplified - in practice would use full pipeline)
            if not candidates:
                gap_class = GapClass.GAP
            else:
                best_score = max(c.bi_encoder_score for c in candidates)
                gap_class = GapClass.GAP if best_score < 0.4 else GapClass.PARTIAL
            
            # Detect unsupported
            unsupported_req = self.detect_unsupported(
                reg_chunk, candidates, gap_class
            )
            
            if unsupported_req:
                unsupported.append(unsupported_req)
        
        logger.info(f"Found {len(unsupported)} unsupported requirements "
                   f"({len(unsupported)/len(all_data['ids'])*100:.1f}% of total)")
        
        return unsupported
    
    def generate_report(
        self,
        unsupported: List[UnsupportedRequirement],
        output_path: str
    ) -> Dict:
        """
        Generate unsupported requirements report
        
        Returns report statistics
        """
        # Sort by severity
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        unsupported.sort(key=lambda x: severity_order.get(x.severity, 99))
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_unsupported': len(unsupported),
                'by_severity': {},
                'by_requirement_type': {},
                'by_detection_method': {}
            },
            'critical_requirements': [],
            'all_requirements': []
        }
        
        for req in unsupported:
            # Update summary
            sev = req.severity
            report['summary']['by_severity'][sev] = \
                report['summary']['by_severity'].get(sev, 0) + 1
            
            req_type = req.requirement_type
            report['summary']['by_requirement_type'][req_type] = \
                report['summary']['by_requirement_type'].get(req_type, 0) + 1
            
            method = req.detection_method
            report['summary']['by_detection_method'][method] = \
                report['summary']['by_detection_method'].get(method, 0) + 1
            
            # Add to appropriate list
            req_dict = req.to_dict()
            if sev == 'critical':
                report['critical_requirements'].append(req_dict)
            report['all_requirements'].append(req_dict)
        
        # Save report
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved unsupported requirements report: {output_path}")
        
        return report['summary']
    
    def get_stats(self) -> Dict:
        """Get detection statistics"""
        return self.detection_stats.copy()


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Day 25: Unsupported Requirements Detector Test")
    print("=" * 60)
    
    detector = UnsupportedRequirementsDetector()
    
    # Test severity calculation
    print("\n1. Testing severity calculation...")
    test_texts = [
        ("Organizations must immediately report data breaches to authorities.", "obligation"),
        ("Employees should consider security best practices.", "recommendation"),
        ("The use of unauthorized software is prohibited.", "prohibition"),
        ("Companies may implement additional safeguards.", "permission"),
    ]
    
    for text, req_type in test_texts:
        severity, score = detector.calculate_severity(text, req_type)
        print(f"   [{req_type:12}] {severity:8} (score: {score:.2f}): {text[:50]}...")
    
    # Test detection
    print("\n2. Testing unsupported detection...")
    
    # Case 1: No candidates
    reg_chunk_no_cands = {
        'chunk_id': 'reg_no_cand',
        'content': 'Organizations must implement quantum encryption by 2025.',
        'metadata': {'source': 'Test Reg', 'page_number': 1}
    }
    result = detector.detect_unsupported(reg_chunk_no_cands, candidates=[], gap_classification=GapClass.GAP)
    if result:
        print(f"   No candidates: {result.severity} severity, {result.detection_method}")
        print(f"     → Priority: {result.recommended_priority}, Effort: {result.estimated_effort}")
    
    # Case 2: Low scores
    from candidate_generation.candidate_generator import CandidatePair
    low_candidates = [
        CandidatePair("reg_low", "text", {}, "pol_1", "text", {}, 0.25, 1),
        CandidatePair("reg_low", "text", {}, "pol_2", "text", {}, 0.20, 2)
    ]
    reg_chunk_low = {
        'chunk_id': 'reg_low',
        'content': 'All systems must use hardware security modules.',
        'metadata': {'source': 'Test Reg', 'page_number': 2}
    }
    result = detector.detect_unsupported(reg_chunk_low, candidates=low_candidates, gap_classification=GapClass.GAP)
    if result:
        print(f"   Low scores: {result.severity} severity, confidence {result.confidence:.2f}")
    
    # Case 3: Supported (should return None)
    reg_chunk_ok = {
        'chunk_id': 'reg_ok',
        'content': 'Organizations should maintain documentation.',
        'metadata': {'source': 'Test Reg', 'page_number': 3}
    }
    good_candidates = [
        CandidatePair("reg_ok", "text", {}, "pol_1", "text", {}, 0.85, 1)
    ]
    result = detector.detect_unsupported(reg_chunk_ok, candidates=good_candidates, gap_classification=None)
    print(f"   Supported case: {'None (correct)' if result is None else 'ERROR - should be None'}")
    
    # Test report generation
    print("\n3. Testing report generation...")
    mock_unsupported = [
        UnsupportedRequirement(
            f"reg_{i}", f"Critical requirement {i}", "Test Source", i+1,
            'critical' if i < 2 else 'high',
            0.8 if i < 2 else 0.6,
            'no_candidates' if i % 2 == 0 else 'low_scores',
            0.9,
            None,
            'obligation',
            'immediate',
            'high'
        )
        for i in range(5)
    ]
    
    summary = detector.generate_report(mock_unsupported, "outputs/day25_unsupported_report.json")
    print(f"   Report summary: {summary}")
    
    # Stats
    print("\n4. Detection statistics:")
    stats = detector.get_stats()
    print(f"   Total checked: {stats['total_checked']}")
    print(f"   Unsupported found: {stats['unsupported_found']}")
    print(f"   By severity: {stats['by_severity']}")
    
    print("\n" + "=" * 60)
    print("=" * 60)