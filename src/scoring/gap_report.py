"""Gap Report Generator - Comprehensive analysis reporting"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import logging
from jsonschema import validate, ValidationError

from scoring.gap_classifier import GapClassification, GapClass
from scoring.unsupported_detector import UnsupportedRequirement

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# JSON Schema for gap report validation
GAP_REPORT_SCHEMA = {
    "type": "object",
    "required": ["report_metadata", "summary", "regulation_analysis"],
    "properties": {
        "report_metadata": {
            "type": "object",
            "required": ["report_id", "generated_at", "version"],
            "properties": {
                "report_id": {"type": "string"},
                "generated_at": {"type": "string"},
                "version": {"type": "string"},
                "tool_version": {"type": "string"},
                "configuration": {"type": "object"}
            }
        },
        "summary": {
            "type": "object",
            "required": ["total_regulations", "total_policy_matches"],
            "properties": {
                "total_regulations": {"type": "integer"},
                "total_policy_matches": {"type": "integer"},
                "classifications": {
                    "type": "object",
                    "properties": {
                        "aligned": {"type": "integer"},
                        "partial": {"type": "integer"},
                        "gap": {"type": "integer"},
                        "unmatched": {"type": "integer"}
                    }
                },
                "confidence_distribution": {
                    "type": "object",
                    "properties": {
                        "high": {"type": "integer"},
                        "medium": {"type": "integer"},
                        "low": {"type": "integer"}
                    }
                }
            }
        },
        "regulation_analysis": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["regulation_chunk_id", "classification", "confidence"],
                "properties": {
                    "regulation_chunk_id": {"type": "string"},
                    "classification": {"type": "string", "enum": ["aligned", "partial", "gap", "unmatched"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "cross_encoder_score": {"type": "number"},
                    "bi_encoder_score": {"type": "number"},
                    "policy_matches": {"type": "array"}
                }
            }
        }
    }
}


@dataclass
class RegulationGapReport:
    """Single regulation chunk gap report"""
    regulation_chunk_id: str
    regulation_text: str
    regulation_source: str
    regulation_page: Optional[int]
    
    classification: str  # aligned, partial, gap, unmatched
    confidence: float
    confidence_level: str
    
    # Scores
    cross_encoder_score: float
    bi_encoder_score: float
    final_score: float
    
    # Matches
    policy_matches: List[Dict]
    num_matches: int
    
    # Gap details (if applicable)
    gap_reason: Optional[str]
    recommended_action: str
    priority: Optional[str]
    
    # Metadata
    processing_time_ms: float
    analyzed_at: str


class GapReportGenerator:
    """
    Generates comprehensive gap analysis reports
    
    Supports:
    - Single chunk reports
    - Full document reports
    - Batch processing reports
    """
    
    def __init__(self, version: str = "1.0.0"):
        self.version = version
        self.report_counter = 0
    
    def _generate_report_id(self) -> str:
        """Generate unique report ID"""
        self.report_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"GAP_{timestamp}_{self.report_counter:04d}"
    
    def create_single_report(
        self,
        classification: GapClassification,
        processing_time_ms: float = 0.0
    ) -> RegulationGapReport:
        """
        Create report for single regulation chunk
        
        Args:
            classification: GapClassification from classifier
            processing_time_ms: Processing time
        
        Returns:
            RegulationGapReport
        """
        return RegulationGapReport(
            regulation_chunk_id=classification.regulation_chunk_id,
            regulation_text=classification.regulation_text,
            regulation_source=classification.regulation_metadata.get('source', 'unknown'),
            regulation_page=classification.regulation_metadata.get('page_number'),
            classification=classification.classification.value,
            confidence=classification.confidence,
            confidence_level=classification.confidence_level,
            cross_encoder_score=classification.cross_encoder_score,
            bi_encoder_score=classification.bi_encoder_score,
            final_score=classification.final_score,
            policy_matches=classification.policy_matches,
            num_matches=len(classification.policy_matches),
            gap_reason=classification.reasoning if classification.classification != GapClass.ALIGNED else None,
            recommended_action=classification.recommended_action,
            priority='high' if classification.classification in [GapClass.GAP, GapClass.UNMATCHED] else 'normal',
            processing_time_ms=processing_time_ms,
            analyzed_at=classification.classified_at
        )
    
    def create_batch_report(
        self,
        classifications: List[GapClassification],
        unsupported: Optional[List[UnsupportedRequirement]] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Create comprehensive batch report
        
        Args:
            classifications: List of gap classifications
            unsupported: Optional list of unsupported requirements
            metadata: Optional report metadata
        
        Returns:
            Complete gap report dict
        """
        report_id = self._generate_report_id()
        generated_at = datetime.now().isoformat()
        
        # Build regulation analysis list
        regulation_analysis = []
        # ── DROP-IN REPLACEMENT for the report_entry block inside create_batch_report ──
# Find this in gap_report.py (around line 140) and replace the entire
# "for cls in classifications:" loop with this version.
#
# WHAT WAS WRONG:
#   - policy_document field was missing entirely
#   - source field was missing
#   - best_score field was missing (UI reads best_score, not final_score)
#   - regulation_text was truncated to 200 chars (too short for UI display)
#   - policy_text was never included (LLM explanation needs it)

        for cls in classifications:
            # Extract best policy match details for easy UI access
            best_match      = cls.policy_matches[0] if cls.policy_matches else {}
            policy_doc      = best_match.get('policy_source') or best_match.get('source') or 'No match found'
            policy_text     = best_match.get('policy_text')  or best_match.get('text')    or ''
            policy_chunk_id = best_match.get('policy_chunk_id') or best_match.get('chunk_id') or ''

            report_entry = {
                # Regulation fields
                'regulation_chunk_id': cls.regulation_chunk_id,
                'regulation_text':     cls.regulation_text,          # full text, not truncated
                'source':              cls.regulation_metadata.get('source', 'unknown'),
                'regulation':          cls.regulation_metadata.get('source', 'unknown'),
                'page_number':         cls.regulation_metadata.get('page_number'),
                'section_header':      cls.regulation_metadata.get('section_header', ''),

                # Classification
                'classification':      cls.classification.value,
                'status':              cls.classification.value,      # alias for UI
                'confidence':          cls.confidence,
                'confidence_level':    cls.confidence_level,

                # Scores — all three names so UI/downstream always finds one
                'best_score':          cls.final_score,
                'final_score':         cls.final_score,
                'cross_encoder_score': cls.cross_encoder_score,
                'bi_encoder_score':    cls.bi_encoder_score,

                # Policy match — top result promoted to top level for easy access
                'policy_document':     policy_doc,
                'matched_policy':      policy_doc,                   # alias
                'policy_text':         policy_text,
                'policy_chunk_id':     policy_chunk_id,
                'policy_matches':      cls.policy_matches,           # full list kept

                # Gap details
                'reasoning':           cls.reasoning,
                'recommended_action':  cls.recommended_action,

                # Explanation placeholder (filled by run_explanations.py)
                'explanation':         '',
            }
            regulation_analysis.append(report_entry)

# ── END REPLACEMENT ────────────────────────────────────────────────────────────
        
        # Calculate summary statistics
        total = len(classifications)
        classifications_count = {
            'aligned': sum(1 for c in classifications if c.classification == GapClass.ALIGNED),
            'partial': sum(1 for c in classifications if c.classification == GapClass.PARTIAL),
            'gap': sum(1 for c in classifications if c.classification == GapClass.GAP),
            'unmatched': sum(1 for c in classifications if c.classification == GapClass.UNMATCHED)
        }
        
        confidence_dist = {
            'high': sum(1 for c in classifications if c.confidence_level == 'high'),
            'medium': sum(1 for c in classifications if c.confidence_level == 'medium'),
            'low': sum(1 for c in classifications if c.confidence_level == 'low')
        }
        
        total_matches = sum(len(c.policy_matches) for c in classifications)
        
        # Build report
        report = {
            'report_metadata': {
                'report_id': report_id,
                'generated_at': generated_at,
                'version': self.version,
                'tool_version': 'regulatory-gap-analyzer-v1',
                'configuration': metadata or {}
            },
            'summary': {
                'total_regulations': total,
                'total_policy_matches': total_matches,
                'classifications': classifications_count,
                'confidence_distribution': confidence_dist,
                'coverage_percentage': round(
                    (classifications_count['aligned'] / total * 100), 1
                ) if total > 0 else 0
            },
            'regulation_analysis': regulation_analysis,
            'unsupported_requirements': [
                req.to_dict() for req in (unsupported or [])
            ],
            'findings': {
                'critical_gaps': [
                    {
                        'regulation_id': c.regulation_chunk_id,
                        'reason': c.reasoning
                    }
                    for c in classifications
                    if c.classification in [GapClass.GAP, GapClass.UNMATCHED] 
                    and c.confidence_level == 'high'
                ],
                'recommendations': self._generate_recommendations(classifications)
            }
        }
        
        return report
    
    def _generate_recommendations(self, classifications: List[GapClassification]) -> List[Dict]:
        """Generate actionable recommendations based on findings"""
        recommendations = []
        
        gap_count = sum(1 for c in classifications if c.classification == GapClass.GAP)
        partial_count = sum(1 for c in classifications if c.classification == GapClass.PARTIAL)
        unmatched_count = sum(1 for c in classifications if c.classification == GapClass.UNMATCHED)
        
        if unmatched_count > 0:
            recommendations.append({
                'priority': 'critical',
                'category': 'missing_coverage',
                'description': f'{unmatched_count} regulations have no policy coverage',
                'action': 'Create new policies to address uncovered requirements'
            })
        
        if gap_count > 0:
            recommendations.append({
                'priority': 'high',
                'category': 'inadequate_coverage',
                'description': f'{gap_count} regulations have inadequate policy coverage',
                'action': 'Update existing policies to address gaps'
            })
        
        if partial_count > 0:
            recommendations.append({
                'priority': 'medium',
                'category': 'partial_coverage',
                'description': f'{partial_count} regulations need review for completeness',
                'action': 'Manual review recommended to verify coverage'
            })
        
        # Check for patterns
        low_conf_aligned = sum(1 for c in classifications 
                              if c.classification == GapClass.ALIGNED and c.confidence_level == 'low')
        if low_conf_aligned > 5:
            recommendations.append({
                'priority': 'low',
                'category': 'model_uncertainty',
                'description': f'{low_conf_aligned} "aligned" classifications have low confidence',
                'action': 'Consider retraining or fine-tuning scoring models'
            })
        
        return recommendations
    
    def validate_report(self, report: Dict) -> Tuple[bool, Optional[str]]:
        """
        Validate report against schema
        
        Returns: (is_valid, error_message)
        """
        try:
            validate(instance=report, schema=GAP_REPORT_SCHEMA)
            return True, None
        except ValidationError as e:
            return False, f"Schema validation failed: {e.message}"
    
    def export_report(
        self,
        report: Dict,
        output_path: str,
        validate_schema: bool = True
    ) -> bool:
        """
        Export report to JSON file
        
        Args:
            report: Report dict
            output_path: Output file path
            validate_schema: Whether to validate before export
        
        Returns:
            Success boolean
        """
        if validate_schema:
            is_valid, error = self.validate_report(report)
            if not is_valid:
                logger.error(f"Report validation failed: {error}")
                return False
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Exported gap report: {output_path}")
        return True
    
    def generate_executive_summary(self, report: Dict) -> str:
        """Generate human-readable executive summary"""
        summary = report['summary']
        
        lines = [
            "=" * 60,
            "GAP ANALYSIS EXECUTIVE SUMMARY",
            "=" * 60,
            f"Report ID: {report['report_metadata']['report_id']}",
            f"Generated: {report['report_metadata']['generated_at']}",
            "",
            "OVERVIEW",
            "-" * 40,
            f"Total Regulations Analyzed: {summary['total_regulations']}",
            f"Total Policy Matches: {summary['total_policy_matches']}",
            f"Overall Coverage: {summary['coverage_percentage']}%",
            "",
            "CLASSIFICATION BREAKDOWN",
            "-" * 40,
            f"   Aligned:     {summary['classifications']['aligned']:4d} ({summary['classifications']['aligned']/summary['total_regulations']*100:.1f}%)",
            f"   Partial:     {summary['classifications']['partial']:4d} ({summary['classifications']['partial']/summary['total_regulations']*100:.1f}%)",
            f"   Gap:         {summary['classifications']['gap']:4d} ({summary['classifications']['gap']/summary['total_regulations']*100:.1f}%)",
            f"   Unmatched:   {summary['classifications']['unmatched']:4d} ({summary['classifications']['unmatched']/summary['total_regulations']*100:.1f}%)",
            "",
            "CONFIDENCE DISTRIBUTION",
            "-" * 40,
            f"  High:    {summary['confidence_distribution']['high']}",
            f"  Medium:  {summary['confidence_distribution']['medium']}",
            f"  Low:     {summary['confidence_distribution']['low']}",
            "",
            "RECOMMENDATIONS",
            "-" * 40,
        ]
        
        for rec in report['findings']['recommendations']:
            lines.append(f"  [{rec['priority'].upper()}] {rec['category']}: {rec['action']}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Day 26: Gap Report Generator Test")
    print("=" * 60)
    
    generator = GapReportGenerator(version="1.0.0")
    
    # Create mock classifications
    print("\n1. Creating mock classifications...")
    mock_classifications = []
    for i in range(10):
        cls_type = [GapClass.ALIGNED, GapClass.PARTIAL, GapClass.GAP, GapClass.UNMATCHED][i % 4]
        confidence = 0.9 if i % 3 == 0 else (0.7 if i % 3 == 1 else 0.5)
        
        cls = GapClassification(
            regulation_chunk_id=f"reg_{i:03d}",
            regulation_text=f"Sample regulation requirement {i}",
            regulation_metadata={'source': f'Regulation Doc {i//5}', 'page_number': i+1},
            classification=cls_type,
            confidence=confidence,
            confidence_level='high' if confidence > 0.8 else ('medium' if confidence > 0.6 else 'low'),
            bi_encoder_score=0.4 + (i * 0.05),
            cross_encoder_score=0.5 + (i * 0.04),
            final_score=0.45 + (i * 0.045),
            threshold_min=0.0,
            threshold_max=1.0,
            reasoning=f"Sample reasoning for {cls_type.value}",
            recommended_action="Sample action",
            policy_matches=[{'policy_id': f'pol_{i}', 'score': 0.8}] if cls_type != GapClass.UNMATCHED else [],
            classified_at=datetime.now().isoformat(),
            config_version="20240301"
        )
        mock_classifications.append(cls)
    
    print(f"   Created {len(mock_classifications)} mock classifications")
    
    # Create batch report
    print("\n2. Creating batch report...")
    report = generator.create_batch_report(
        classifications=mock_classifications,
        unsupported=[],
        metadata={'test_run': True, 'thresholds': {'aligned': 0.7, 'partial': 0.4}}
    )
    
    print(f"   Report ID: {report['report_metadata']['report_id']}")
    print(f"   Total regulations: {report['summary']['total_regulations']}")
    print(f"   Classifications: {report['summary']['classifications']}")
    
    # Validate report
    print("\n3. Validating report schema...")
    is_valid, error = generator.validate_report(report)
    print(f"   Valid: {' Yes' if is_valid else ' No'}")
    if error:
        print(f"   Error: {error}")
    
    # Export
    print("\n4. Exporting report...")
    success = generator.export_report(report, "outputs/day26_gap_report.json")
    print(f"   Export: {' Success' if success else ' Failed'}")
    
    # Executive summary
    print("\n5. Executive Summary:")
    print(generator.generate_executive_summary(report))
    
    print("\n" + "=" * 60)
    print("=" * 60)