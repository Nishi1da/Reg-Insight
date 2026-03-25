"""Prompt Quality Analyzer"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Optional
import json
import logging
from datetime import datetime
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Quality score for a single explanation"""
    explanation_id: str
    accuracy_score: float       # 0-1: factual, not hallucinated
    actionability_score: float  # 0-1: recommendation is specific
    conciseness_score: float    # 0-1: not too long or too short
    structure_score: float      # 0-1: valid JSON, all fields present
    risk_consistency_score: float  # 0-1: risk level matches content
    overall_score: float        # weighted average
    issues: List[str]           # list of problems found
    passed: bool                # overall pass/fail


class PromptQualityAnalyzer:
    """
    Analyzes quality of generated explanations
    
    Evaluates:
    - Accuracy (no hallucinations)
    - Actionability (specific recommendations)
    - Conciseness (right length)
    - Structure (valid JSON)
    - Risk consistency (risk matches content)
    """

    def __init__(self):
        self.rubric = {
            'accuracy': {
                'weight': 0.30,
                'description': 'Factual, based on provided text only',
                'checks': [
                    'no_invented_clauses',
                    'references_provided_text',
                    'no_contradictions'
                ]
            },
            'actionability': {
                'weight': 0.25,
                'description': 'Recommendation is specific and doable',
                'checks': [
                    'has_specific_action',
                    'not_vague',
                    'mentions_policy_type'
                ]
            },
            'conciseness': {
                'weight': 0.20,
                'description': 'Right length, no padding',
                'checks': [
                    'summary_length_ok',
                    'not_repetitive',
                    'recommendation_specific'
                ]
            },
            'structure': {
                'weight': 0.15,
                'description': 'All required fields present',
                'checks': [
                    'all_fields_present',
                    'valid_enum_values',
                    'list_not_empty'
                ]
            },
            'risk_consistency': {
                'weight': 0.10,
                'description': 'Risk level matches gap severity',
                'checks': [
                    'risk_matches_language',
                    'not_all_critical',
                    'not_all_low'
                ]
            }
        }

        self.required_fields = [
            'summary', 'recommendation', 'risk_level',
            'key_differences', 'confidence'
        ]

        self.valid_risk_levels = [
            'low', 'medium', 'high', 'critical'
        ]

        self.valid_confidence_levels = [
            'low', 'medium', 'high'
        ]

        # Words suggesting vague recommendations
        self.vague_words = [
            'consider', 'might', 'could', 'perhaps',
            'review', 'look into', 'think about'
        ]

        # Words suggesting specific actions
        self.action_words = [
            'implement', 'create', 'update', 'add',
            'establish', 'define', 'require', 'mandate',
            'document', 'enforce', 'develop', 'write'
        ]

        self.stats = {
            'total_analyzed': 0,
            'passed': 0,
            'failed': 0,
            'common_issues': {}
        }

    def analyze(
        self,
        explanation: Dict,
        regulation_text: str = "",
        policy_text: str = ""
    ) -> QualityScore:
        """
        Analyze a single explanation

        Args:
            explanation: The explanation dict from LLM
            regulation_text: Original regulation text
            policy_text: Original policy text

        Returns:
            QualityScore with detailed breakdown
        """
        self.stats['total_analyzed'] += 1
        issues = []

        # ── Score each dimension ──
        accuracy = self._score_accuracy(
            explanation, regulation_text, issues
        )
        actionability = self._score_actionability(
            explanation, issues
        )
        conciseness = self._score_conciseness(
            explanation, issues
        )
        structure = self._score_structure(
            explanation, issues
        )
        risk_consistency = self._score_risk_consistency(
            explanation, issues
        )

        # ── Weighted overall score ──
        overall = (
            accuracy * 0.30
            + actionability * 0.25
            + conciseness * 0.20
            + structure * 0.15
            + risk_consistency * 0.10
        )

        passed = overall >= 0.70 and structure >= 0.80

        if passed:
            self.stats['passed'] += 1
        else:
            self.stats['failed'] += 1

        for issue in issues:
            self.stats['common_issues'][issue] = (
                self.stats['common_issues'].get(issue, 0) + 1
            )

        return QualityScore(
            explanation_id=explanation.get(
                'regulation_chunk_id', 'unknown'
            ),
            accuracy_score=accuracy,
            actionability_score=actionability,
            conciseness_score=conciseness,
            structure_score=structure,
            risk_consistency_score=risk_consistency,
            overall_score=round(overall, 3),
            issues=issues,
            passed=passed
        )

    def _score_accuracy(
        self,
        exp: Dict,
        reg_text: str,
        issues: List
    ) -> float:
        """Check for hallucinations and accuracy"""
        score = 1.0
        explanation_data = exp.get('explanation', exp)

        summary = str(explanation_data.get('summary', ''))
        recommendation = str(
            explanation_data.get('recommendation', '')
        )

        # Check for invented article numbers
        import re
        articles = re.findall(
            r'article\s+\d+|section\s+\d+|clause\s+\d+',
            (summary + recommendation).lower()
        )
        if articles and reg_text:
            for article in articles:
                if article not in reg_text.lower():
                    score -= 0.3
                    issues.append('invented_clause_reference')
                    break

        # Check summary references provided content
        if reg_text and len(reg_text) > 20:
            reg_words = set(reg_text.lower().split())
            summary_words = set(summary.lower().split())
            overlap = len(
                reg_words.intersection(summary_words)
            ) / max(len(reg_words), 1)
            if overlap < 0.05:
                score -= 0.2
                issues.append('summary_not_based_on_regulation')

        return max(0.0, score)

    def _score_actionability(
        self,
        exp: Dict,
        issues: List
    ) -> float:
        """Check if recommendation is specific"""
        score = 1.0
        explanation_data = exp.get('explanation', exp)
        rec = str(
            explanation_data.get('recommendation', '')
        ).lower()

        if not rec or len(rec) < 20:
            score -= 0.5
            issues.append('recommendation_too_short')
            return max(0.0, score)

        # Check for vague language
        vague_count = sum(
            1 for w in self.vague_words if w in rec
        )
        if vague_count >= 2:
            score -= 0.3
            issues.append('recommendation_too_vague')

        # Check for action words
        has_action = any(w in rec for w in self.action_words)
        if not has_action:
            score -= 0.3
            issues.append('no_specific_action_in_recommendation')

        return max(0.0, score)

    def _score_conciseness(
        self,
        exp: Dict,
        issues: List
    ) -> float:
        """Check length is appropriate"""
        score = 1.0
        explanation_data = exp.get('explanation', exp)

        summary = str(explanation_data.get('summary', ''))
        rec = str(explanation_data.get('recommendation', ''))

        # Summary length check
        if len(summary) < 20:
            score -= 0.4
            issues.append('summary_too_short')
        elif len(summary) > 200:
            score -= 0.2
            issues.append('summary_too_long')

        # Recommendation length check
        if len(rec) < 30:
            score -= 0.3
            issues.append('recommendation_too_short')
        elif len(rec) > 500:
            score -= 0.1
            issues.append('recommendation_too_long')

        # Check for repetition
        if summary and rec:
            if summary[:50].lower() in rec.lower():
                score -= 0.2
                issues.append('summary_repeated_in_recommendation')

        return max(0.0, score)

    def _score_structure(
        self,
        exp: Dict,
        issues: List
    ) -> float:
        """Check all required fields present"""
        score = 1.0
        explanation_data = exp.get('explanation', exp)

        # Check required fields
        for field in self.required_fields:
            if field not in explanation_data:
                score -= 0.2
                issues.append(f'missing_field_{field}')

        # Check risk level is valid
        risk = explanation_data.get('risk_level', '')
        if hasattr(risk, 'value'):
            risk = risk.value
            risk = str(risk).lower()

        # Check confidence is valid
        conf =explanation_data.get('confidence', '')
        if hasattr(conf, 'value'):
            conf = conf.value
            conf = str(conf).lower()

        # Check key_differences is non-empty list
        diffs = explanation_data.get('key_differences', [])
        if not isinstance(diffs, list) or len(diffs) == 0:
            score -= 0.2
            issues.append('empty_key_differences')

        return max(0.0, score)

    def _score_risk_consistency(
        self,
        exp: Dict,
        issues: List
    ) -> float:
        """Check risk level matches content"""
        score = 1.0
        explanation_data = exp.get('explanation', exp)

        risk = explanation_data.get('risk_level', '')
        if hasattr(risk, 'value'):
            risk = risk.value
            risk = str(risk).lower()
        summary = str(
            explanation_data.get('summary', '')
        ).lower()
        rec = str(
            explanation_data.get('recommendation', '')
        ).lower()
        combined = summary + ' ' + rec

        # Critical risk should mention serious issues
        critical_words = [
            'critical', 'severe', 'immediately',
            'urgent', 'serious', 'major'
        ]
        low_words = [
            'minor', 'small', 'minimal',
            'slight', 'basic'
        ]

        if risk == 'critical':
            has_critical_language = any(
                w in combined for w in critical_words
            )
            if not has_critical_language:
                score -= 0.3
                issues.append(
                    'critical_risk_without_critical_language'
                )

        if risk == 'low':
            has_serious_language = any(
                w in combined
                for w in ['critical', 'severe', 'major']
            )
            if has_serious_language:
                score -= 0.4
                issues.append(
                    'low_risk_with_serious_language'
                )

        return max(0.0, score)

    def analyze_batch(
        self,
        explanations: List[Dict],
        regulation_texts: Optional[List[str]] = None,
        policy_texts: Optional[List[str]] = None
    ) -> Dict:
        """Analyze multiple explanations"""
        scores = []

        for i, exp in enumerate(explanations):
            reg_text = (
                regulation_texts[i]
                if regulation_texts and i < len(regulation_texts)
                else ""
            )
            pol_text = (
                policy_texts[i]
                if policy_texts and i < len(policy_texts)
                else ""
            )
            score = self.analyze(exp, reg_text, pol_text)
            scores.append(score)

        return self._generate_report(scores)

    def _generate_report(
        self,
        scores: List[QualityScore]
    ) -> Dict:
        """Generate quality analysis report"""
        if not scores:
            return {}

        total = len(scores)
        passed = sum(1 for s in scores if s.passed)

        avg_scores = {
            'accuracy': sum(
                s.accuracy_score for s in scores
            ) / total,
            'actionability': sum(
                s.actionability_score for s in scores
            ) / total,
            'conciseness': sum(
                s.conciseness_score for s in scores
            ) / total,
            'structure': sum(
                s.structure_score for s in scores
            ) / total,
            'risk_consistency': sum(
                s.risk_consistency_score for s in scores
            ) / total,
            'overall': sum(
                s.overall_score for s in scores
            ) / total
        }

        # Find most common issues
        all_issues = {}
        for score in scores:
            for issue in score.issues:
                all_issues[issue] = (
                    all_issues.get(issue, 0) + 1
                )

        top_issues = sorted(
            all_issues.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        return {
            'total_analyzed': total,
            'passed': passed,
            'failed': total - passed,
            'pass_rate': round(passed / total * 100, 1),
            'average_scores': {
                k: round(v, 3)
                for k, v in avg_scores.items()
            },
            'top_issues': top_issues,
            'weakest_dimension': min(
                avg_scores,
                key=avg_scores.get
            ),
            'strongest_dimension': max(
                avg_scores,
                key=avg_scores.get
            ),
            'generated_at': datetime.now().isoformat(),
            'individual_scores': [
                {
                    'id': s.explanation_id,
                    'overall': s.overall_score,
                    'passed': s.passed,
                    'issues': s.issues
                }
                for s in scores
            ]
        }

    def get_improvement_suggestions(
        self,
        report: Dict
    ) -> List[str]:
        """Generate prompt improvement suggestions"""
        suggestions = []
        avg = report.get('average_scores', {})

        if avg.get('accuracy', 1) < 0.7:
            suggestions.append(
                "Add instruction: 'Base analysis ONLY on "
                "the provided regulation and policy text. "
                "Do not reference external information.'"
            )

        if avg.get('actionability', 1) < 0.7:
            suggestions.append(
                "Add instruction: 'Recommendation must "
                "start with an action verb (implement, "
                "create, update, add, establish).'"
            )

        if avg.get('conciseness', 1) < 0.7:
            suggestions.append(
                "Add instruction: 'Summary must be "
                "1-2 sentences maximum. Recommendation "
                "must be under 100 words.'"
            )

        if avg.get('structure', 1) < 0.8:
            suggestions.append(
                "Add few-shot example showing correct "
                "JSON structure with all required fields."
            )

        if avg.get('risk_consistency', 1) < 0.7:
            suggestions.append(
                "Add risk level definitions: "
                "critical=immediate legal action risk, "
                "high=significant compliance gap, "
                "medium=partial coverage, "
                "low=minor improvement needed"
            )

        return suggestions

    def get_stats(self) -> Dict:
        """Get analyzer statistics"""
        total = self.stats['total_analyzed']
        return {
            **self.stats,
            'pass_rate': round(
                self.stats['passed'] / total * 100
                if total > 0 else 0, 1
            )
        }


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Prompt Quality Analyzer Test")
    print("=" * 60)

    analyzer = PromptQualityAnalyzer()

    test_explanations = [
        {
            'regulation_chunk_id': 'reg_001',
            'explanation': {
                'summary': (
                    'Policy lacks MFA requirement '
                    'for admin access.'
                ),
                'recommendation': (
                    'Implement multi-factor authentication '
                    'for all admin accounts using TOTP '
                    'or hardware keys.'
                ),
                'risk_level': 'high',
                'key_differences': [
                    'MFA not mentioned in policy',
                    'Only passwords covered'
                ],
                'confidence': 'high'
            }
        },
        {
            'regulation_chunk_id': 'reg_002',
            'explanation': {
                'summary': 'Gap.',
                'recommendation': 'Consider reviewing.',
                'risk_level': 'critical',
                'key_differences': [],
                'confidence': 'high'
            }
        },
        {
            'regulation_chunk_id': 'reg_003',
            'explanation': {
                'summary': (
                    'The organization should consider '
                    'looking into the possibility of '
                    'perhaps implementing some form of '
                    'data protection that might be '
                    'suitable for their needs.'
                ),
                'recommendation': (
                    'Update data protection policy to '
                    'include encryption requirements.'
                ),
                'risk_level': 'medium',
                'key_differences': ['Missing encryption'],
                'confidence': 'medium'
            }
        }
    ]

    print("\n1. Analyzing 3 test explanations...")
    for exp in test_explanations:
        score = analyzer.analyze(exp)
        status = " PASS" if score.passed else " FAIL"
        print(
            f"   {status} {score.explanation_id}: "
            f"{score.overall_score:.2f} "
            f"{'Issues: ' + str(score.issues) if score.issues else ''}"
        )

    print("\n2. Batch analysis report...")
    report = analyzer.analyze_batch(test_explanations)
    print(f"   Pass rate: {report['pass_rate']}%")
    print(
        f"   Average overall: "
        f"{report['average_scores']['overall']:.2f}"
    )
    print(f"   Weakest: {report['weakest_dimension']}")
    print(f"   Top issues: {report['top_issues'][:3]}")

    print("\n3. Improvement suggestions...")
    suggestions = analyzer.get_improvement_suggestions(report)
    for i, s in enumerate(suggestions, 1):
        print(f"   {i}. {s[:80]}...")

    print("\n" + "=" * 60)
    print("=" * 60)