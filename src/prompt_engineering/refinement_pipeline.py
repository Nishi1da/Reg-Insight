"""Day 40: Output Refinement Pipeline"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List, Optional, Tuple
import re
import logging
from datetime import datetime
from dataclasses import dataclass

from explanation.groq_client import GroqLLMClient
from explanation.response_parser import LLMResponseParser
from prompt_engineering.quality_analyzer import PromptQualityAnalyzer
from prompt_engineering.structure_improver import StructureImprover

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RefinedOutput:
    """Result of refinement pipeline"""
    original: Dict
    refined: Dict
    quality_score: float
    fixes_applied: List[str]
    passed_validation: bool
    refinement_time_ms: float


class RefinementPipeline:
    """
    Post-processing pipeline for LLM outputs

    Steps:
    1. Text cleaning (whitespace, capitalization)
    2. Terminology standardization
    3. Structure validation and auto-fix
    4. Quality scoring
    5. Explanation ranking (if multiple candidates)
    6. Contradiction check
    """

    def __init__(self):
        self.groq_client = GroqLLMClient()
        self.parser = LLMResponseParser()
        self.quality_analyzer = PromptQualityAnalyzer()
        self.structure_improver = StructureImprover()

        # Compliance terminology standardization
        self.terminology_map = {
            'mfa': 'Multi-Factor Authentication (MFA)',
            'multi factor': 'Multi-Factor Authentication (MFA)',
            '2fa': 'Two-Factor Authentication (2FA)',
            'gdpr': 'GDPR',
            'pii': 'Personally Identifiable Information (PII)',
            'phi': 'Protected Health Information (PHI)',
            'soc2': 'SOC 2',
            'iso27001': 'ISO 27001',
            'aes': 'AES encryption',
            'ssl': 'SSL/TLS',
            'tls': 'SSL/TLS'
        }

        # Templates for common gap types
        self.gap_templates = {
            'authentication': {
                'risk_level': 'high',
                'recommendation_prefix': (
                    'Implement and document authentication '
                    'requirements including'
                )
            },
            'encryption': {
                'risk_level': 'high',
                'recommendation_prefix': (
                    'Define encryption standards and '
                    'implement controls for'
                )
            },
            'data_retention': {
                'risk_level': 'medium',
                'recommendation_prefix': (
                    'Create a data retention policy that '
                    'specifies'
                )
            },
            'access_control': {
                'risk_level': 'high',
                'recommendation_prefix': (
                    'Establish access control procedures '
                    'that define'
                )
            },
            'incident_response': {
                'risk_level': 'high',
                'recommendation_prefix': (
                    'Develop an incident response plan '
                    'that includes'
                )
            }
        }

        self.stats = {
            'total_refined': 0,
            'fixes_applied': 0,
            'passed': 0,
            'failed': 0
        }

    def refine(
        self,
        explanation: Dict,
        regulation_text: str = "",
        policy_text: str = ""
    ) -> RefinedOutput:
        """
        Run full refinement pipeline on explanation

        Args:
            explanation: Raw explanation from LLM
            regulation_text: Original regulation
            policy_text: Original policy

        Returns:
            RefinedOutput with cleaned explanation
        """
        import time
        start = time.time()
        self.stats['total_refined'] += 1
        fixes_applied = []

        # Get explanation data
        exp_data = explanation.get('explanation', explanation)
        refined = exp_data.copy()

        # ── Step 1: Text cleaning ──
        refined, text_fixes = self._clean_text(refined)
        fixes_applied.extend(text_fixes)

        # ── Step 2: Terminology standardization ──
        refined, term_fixes = self._standardize_terminology(
            refined
        )
        fixes_applied.extend(term_fixes)

        # ── Step 3: Structure validation ──
        struct_result = self.structure_improver.check_structure(
            refined
        )
        if struct_result.auto_fixed and struct_result.fixed_data:
            refined = struct_result.fixed_data
            fixes_applied.append('structure_auto_fixed')

        # ── Step 4: Quality scoring ──
        quality = self.quality_analyzer.analyze(
            {'explanation': refined},
            regulation_text,
            policy_text
        )

        # ── Step 5: Contradiction check ──
        refined, contra_fixes = self._check_contradictions(
            refined, regulation_text
        )
        fixes_applied.extend(contra_fixes)

        elapsed = (time.time() - start) * 1000

        if fixes_applied:
            self.stats['fixes_applied'] += len(fixes_applied)

        if quality.passed:
            self.stats['passed'] += 1
        else:
            self.stats['failed'] += 1

        return RefinedOutput(
            original=exp_data,
            refined=refined,
            quality_score=quality.overall_score,
            fixes_applied=fixes_applied,
            passed_validation=quality.passed,
            refinement_time_ms=elapsed
        )

    def _clean_text(
        self,
        data: Dict
    ) -> Tuple[Dict, List[str]]:
        """Clean whitespace and fix capitalization"""
        cleaned = data.copy()
        fixes = []

        text_fields = ['summary', 'recommendation']

        for field in text_fields:
            if field not in cleaned:
                continue

            original = str(cleaned[field])
            value = original.strip()

            # Fix capitalization
            if value and value[0].islower():
                value = value[0].upper() + value[1:]
                fixes.append(f'capitalization_{field}')

            # Fix double spaces
            value = re.sub(r'\s+', ' ', value)

            # Ensure ends with period
            if value and not value.endswith(('.', '!', '?')):
                value += '.'
                fixes.append(f'added_period_{field}')

            if value != original:
                cleaned[field] = value

        # Clean key_differences
        if 'key_differences' in cleaned:
            diffs = cleaned['key_differences']
            if isinstance(diffs, list):
                cleaned_diffs = []
                for diff in diffs:
                    d = str(diff).strip()
                    if d and len(d) >= 5:
                        if d[0].islower():
                            d = d[0].upper() + d[1:]
                        cleaned_diffs.append(d)
                if cleaned_diffs != diffs:
                    cleaned['key_differences'] = cleaned_diffs
                    fixes.append('cleaned_key_differences')

        return cleaned, fixes

    def _standardize_terminology(
        self,
        data: Dict
    ) -> Tuple[Dict, List[str]]:
        """Standardize compliance terminology"""
        standardized = data.copy()
        fixes = []

        text_fields = ['summary', 'recommendation']

        for field in text_fields:
            if field not in standardized:
                continue

            text = str(standardized[field])
            original = text

            for term, standard in self.terminology_map.items():
                # Replace only standalone terms
                pattern = r'\b' + re.escape(term) + r'\b'
                if re.search(pattern, text, re.IGNORECASE):
                    # Only replace if not already standardized
                    if standard.lower() not in text.lower():
                        text = re.sub(
                            pattern, standard, text,
                            flags=re.IGNORECASE
                        )

            if text != original:
                standardized[field] = text
                fixes.append(f'terminology_{field}')

        return standardized, fixes

    def _check_contradictions(
        self,
        data: Dict,
        regulation_text: str
    ) -> Tuple[Dict, List[str]]:
        """Check for contradictions with source text"""
        checked = data.copy()
        fixes = []

        # Check risk level against key_differences count
        risk = checked.get('risk_level', '')
        if hasattr(risk, 'value'):
            risk = risk.value
            risk = str(risk).lower()
            
        diffs = checked.get('key_differences', [])

        # If critical but only 1 minor difference, downgrade
        if (risk == 'critical'
                and len(diffs) == 1
                and len(str(diffs[0])) < 30):
            checked['risk_level'] = 'high'
            fixes.append('downgraded_risk_level')

        # If low but many differences, upgrade
        if risk == 'low' and len(diffs) >= 4:
            checked['risk_level'] = 'medium'
            fixes.append('upgraded_risk_level')

        return checked, fixes

    def rank_candidates(
        self,
        candidates: List[Dict],
        regulation_text: str = "",
        policy_text: str = ""
    ) -> Tuple[Dict, List[Dict]]:
        """
        Rank multiple candidate explanations

        Returns: (best_candidate, ranked_list)
        """
        if not candidates:
            return {}, []

        if len(candidates) == 1:
            return candidates[0], candidates

        scored = []
        for candidate in candidates:
            quality = self.quality_analyzer.analyze(
                {'explanation': candidate},
                regulation_text,
                policy_text
            )
            scored.append({
                'candidate': candidate,
                'score': quality.overall_score,
                'passed': quality.passed
            })

        scored.sort(key=lambda x: x['score'], reverse=True)

        return (
            scored[0]['candidate'],
            [s['candidate'] for s in scored]
        )

    def refine_batch(
        self,
        explanations: List[Dict],
        regulation_texts: Optional[List[str]] = None,
        policy_texts: Optional[List[str]] = None
    ) -> List[RefinedOutput]:
        """Refine a batch of explanations"""
        results = []

        for i, exp in enumerate(explanations):
            reg = (
                regulation_texts[i]
                if regulation_texts and i < len(regulation_texts)
                else ""
            )
            pol = (
                policy_texts[i]
                if policy_texts and i < len(policy_texts)
                else ""
            )
            result = self.refine(exp, reg, pol)
            results.append(result)

        return results

    def get_stats(self) -> Dict:
        """Get pipeline statistics"""
        total = self.stats['total_refined']
        return {
            **self.stats,
            'pass_rate': round(
                self.stats['passed'] / total * 100
                if total > 0 else 0, 1
            ),
            'avg_fixes_per_explanation': round(
                self.stats['fixes_applied'] / total
                if total > 0 else 0, 1
            )
        }


# Test
if __name__ == "__main__":
    print("=" * 60)
    print(" Refinement Pipeline Test")
    print("=" * 60)

    pipeline = RefinementPipeline()

    test_cases = [
        {
            'name': 'Needs cleaning',
            'exp': {
                'summary': (
                    '  policy lacks mfa requirement  '
                ),
                'recommendation': (
                    'implement multi factor '
                    'authentication for admins'
                ),
                'risk_level': 'high',
                'key_differences': [
                    'missing mfa',
                    'passwords only'
                ],
                'confidence': 'high'
            },
            'reg': 'Organizations must implement MFA.'
        },
        {
            'name': 'Needs terminology fix',
            'exp': {
                'summary': (
                    'Policy missing aes encryption '
                    'for pii data.'
                ),
                'recommendation': (
                    'Implement ssl for data in transit '
                    'and aes for storage.'
                ),
                'risk_level': 'high',
                'key_differences': [
                    'No aes mentioned',
                    'pii not addressed'
                ],
                'confidence': 'high'
            },
            'reg': 'Encrypt PII using AES-256.'
        },
        {
            'name': 'Needs risk adjustment',
            'exp': {
                'summary': (
                    'Minor gap found in policy.'
                ),
                'recommendation': (
                    'Update policy to add requirement.'
                ),
                'risk_level': 'critical',
                'key_differences': ['Small gap'],
                'confidence': 'medium'
            },
            'reg': 'Policy should mention backups.'
        }
    ]

    print("\n1. Running refinement pipeline...")
    for case in test_cases:
        result = pipeline.refine(
            case['exp'],
            regulation_text=case['reg']
        )
        status = "PASS" if result.passed_validation else "ERROR"
        print(f"   {status} {case['name']:25}")
        print(
            f"      Score: {result.quality_score:.2f} | "
            f"Fixes: {result.fixes_applied}"
        )
        if result.refined.get('summary') != case['exp'].get(
            'summary'
        ):
            print(
                f"      Before: '{case['exp']['summary'][:50]}'"
            )
            print(
                f"      After:  "
                f"'{result.refined['summary'][:50]}'"
            )

    print("\n2. Ranking candidates test...")
    candidates = [
        {
            'summary': 'Minor gap found.',
            'recommendation': 'Review.',
            'risk_level': 'low',
            'key_differences': ['gap'],
            'confidence': 'low'
        },
        {
            'summary': (
                'Policy lacks encryption requirements '
                'for data at rest.'
            ),
            'recommendation': (
                'Implement AES-256 encryption for all '
                'stored data.'
            ),
            'risk_level': 'high',
            'key_differences': [
                'No encryption standard defined',
                'Storage requirements missing'
            ],
            'confidence': 'high'
        }
    ]
    best, ranked = pipeline.rank_candidates(candidates)
    print(
        f"   Best candidate risk: "
        f"{best.get('risk_level')}"
    )
    print(f"   Rankings: {[c.get('risk_level') for c in ranked]}")

    print("\n3. Stats:")
    stats = pipeline.get_stats()
    print(f"   Pass rate: {stats['pass_rate']}%")
    print(
        f"   Avg fixes per explanation: "
        f"{stats['avg_fixes_per_explanation']}"
    )

    print("\n" + "=" * 60)
    print("=" * 60)