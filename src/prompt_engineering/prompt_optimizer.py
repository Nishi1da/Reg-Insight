"""Prompt Optimizer - Accuracy Improvements"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List, Optional, Tuple
import json
import logging
import time
from datetime import datetime
from dataclasses import dataclass

from explanation.groq_client import GroqLLMClient
from explanation.response_parser import LLMResponseParser
from prompt_engineering.quality_analyzer import (
    PromptQualityAnalyzer, QualityScore
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PromptVariant:
    """A prompt variant for A/B testing"""
    version: str
    name: str
    system_prompt: str
    user_template: str
    description: str
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class PromptOptimizer:
    """
    Iteratively improves prompts for better accuracy

    Features:
    - Few-shot examples
    - Accuracy constraints
    - A/B testing
    - Version tracking
    """

    def __init__(self):
        self.groq_client = GroqLLMClient()
        self.parser = LLMResponseParser()
        self.analyzer = PromptQualityAnalyzer()
        self.variants: Dict[str, PromptVariant] = {}
        self._register_variants()

    def _register_variants(self):
        """Register prompt variants for testing"""

        # ── VARIANT A: Original (Week 5 baseline) ──
        self.variants['v1_baseline'] = PromptVariant(
            version="1.0",
            name="Baseline",
            system_prompt=(
                "You are a senior regulatory compliance analyst.\n"
                "Always respond with valid JSON only.\n"
                "Be specific, factual, and actionable."
            ),
            user_template="""Analyze this regulatory compliance gap.

REGULATION: {regulation_text}
POLICY: {policy_text}
SCORE: {final_score:.2f}/1.0

Respond ONLY with JSON:
{{
    "summary": "gap assessment",
    "recommendation": "action to take",
    "risk_level": "low|medium|high|critical",
    "key_differences": ["gap 1", "gap 2"],
    "confidence": "high|medium|low"
}}""",
            description="Original Week 5 prompt"
        )

        # ── VARIANT B: Accuracy-focused ──
        self.variants['v2_accuracy'] = PromptVariant(
            version="2.0",
            name="Accuracy Focused",
            system_prompt=(
                "You are a senior regulatory compliance analyst.\n"
                "CRITICAL RULES:\n"
                "1. Base your analysis ONLY on the provided texts\n"
                "2. Do NOT reference external regulations or standards\n"
                "3. Do NOT invent clause numbers or article references\n"
                "4. If information is not in the text, say so\n"
                "5. Always respond with valid JSON only"
            ),
            user_template="""Analyze this regulatory compliance gap.
Base your analysis ONLY on the texts provided below.

REGULATION REQUIREMENT:
{regulation_text}

COMPANY POLICY:
{policy_text}

MATCH SCORE: {final_score:.2f}/1.0
(0.0 = no match, 1.0 = perfect match)

Analyze what is present and missing, then respond with JSON:
{{
    "summary": "1-2 sentence gap assessment based on texts above",
    "recommendation": "Specific action starting with a verb",
    "risk_level": "low|medium|high|critical",
    "key_differences": [
        "Specific thing in regulation not in policy",
        "Specific requirement missing from policy"
    ],
    "confidence": "high|medium|low"
}}""",
            description="Reduced hallucinations with constraints"
        )

        # ── VARIANT C: Few-shot with examples ──
        self.variants['v3_fewshot'] = PromptVariant(
            version="3.0",
            name="Few-Shot Examples",
            system_prompt=(
                "You are a senior regulatory compliance analyst.\n"
                "Base analysis ONLY on provided texts.\n"
                "Always respond with valid JSON only.\n"
                "Follow the exact format shown in examples."
            ),
            user_template="""Analyze regulatory compliance gaps.
Base your analysis ONLY on the texts provided.

EXAMPLE 1 (for reference):
Regulation: "All admin accounts must use multi-factor authentication"
Policy: "Users must use strong passwords of 12 characters minimum"
Score: 0.28
Output: {{
    "summary": "Policy only covers passwords, missing MFA requirement entirely.",
    "recommendation": "Implement MFA for all admin accounts using TOTP or hardware keys.",
    "risk_level": "high",
    "key_differences": [
        "Regulation requires MFA, policy only requires passwords",
        "Admin accounts not specifically addressed in policy"
    ],
    "confidence": "high"
}}

EXAMPLE 2 (for reference):
Regulation: "Personal data must be encrypted at rest and in transit"
Policy: "We use AES-256 for database encryption"
Score: 0.61
Output: {{
    "summary": "Policy covers encryption at rest but missing in-transit requirement.",
    "recommendation": "Add TLS/SSL requirements for data in transit to existing encryption policy.",
    "risk_level": "medium",
    "key_differences": [
        "In-transit encryption not mentioned in policy",
        "Policy only covers database, not all personal data"
    ],
    "confidence": "high"
}}

NOW ANALYZE THIS:
Regulation: {regulation_text}
Policy: {policy_text}
Score: {final_score:.2f}

Respond ONLY with JSON (same format as examples):""",
            description="Few-shot with 2 good examples"
        )

        # ── VARIANT D: Chain of thought ──
        self.variants['v4_cot'] = PromptVariant(
            version="4.0",
            name="Chain of Thought",
            system_prompt=(
                "You are a senior regulatory compliance analyst.\n"
                "Think step by step before concluding.\n"
                "Base analysis ONLY on provided texts.\n"
                "Always respond with valid JSON only."
            ),
            user_template="""Analyze this compliance gap step by step.

REGULATION: {regulation_text}
POLICY: {policy_text}
SCORE: {final_score:.2f}/1.0

Think through this systematically:
1. What does the regulation REQUIRE?
2. What does the policy COVER?
3. What is MISSING from the policy?
4. How SERIOUS is this gap?
5. What is the SPECIFIC FIX?

Then respond ONLY with this JSON:
{{
    "summary": "Gap assessment based on your analysis",
    "recommendation": "Specific fix starting with action verb",
    "risk_level": "low|medium|high|critical",
    "key_differences": [
        "Key gap 1 identified in step 3",
        "Key gap 2 identified in step 3"
    ],
    "confidence": "high|medium|low"
}}""",
            description="Chain of thought reasoning"
        )

    def test_variant(
        self,
        variant_name: str,
        test_cases: List[Dict],
        delay_seconds: float = 2.0
    ) -> Dict:
        """
        Test a prompt variant on test cases

        Args:
            variant_name: Which variant to test
            test_cases: List of regulation/policy pairs
            delay_seconds: Delay between API calls

        Returns:
            Quality report for this variant
        """
        if variant_name not in self.variants:
            raise ValueError(f"Variant '{variant_name}' not found")

        variant = self.variants[variant_name]
        responses = []
        latencies = []

        logger.info(
            f"Testing variant: {variant.name} "
            f"on {len(test_cases)} cases"
        )

        for i, case in enumerate(test_cases):
            try:
                # Format prompt
                prompt = variant.user_template.format(
                    regulation_text=case.get(
                        'regulation_text', 'No regulation'
                    ),
                    policy_text=case.get(
                        'policy_text', 'No policy'
                    ),
                    final_score=case.get('final_score', 0.5),
                    cross_encoder_score=case.get(
                        'cross_encoder_score', 0.5
                    ),
                    bi_encoder_score=case.get(
                        'bi_encoder_score', 0.5
                    ),
                    regulation_source=case.get(
                        'regulation_source', 'Unknown'
                    ),
                    policy_source=case.get(
                        'policy_source', 'Unknown'
                    ),
                    classification=case.get(
                        'classification', 'gap'
                    ),
                    confidence=case.get('confidence', 0.7)
                )

                start = time.time()
                response = self.groq_client.generate(
                    prompt=prompt,
                    system_prompt=variant.system_prompt,
                    temperature=0.1
                )
                latency = (time.time() - start) * 1000
                latencies.append(latency)

                parse_result = self.parser.parse(response.text)

                if parse_result.success:
                    responses.append({
                        'regulation_chunk_id': case.get(
                            'id', f'case_{i}'
                        ),
                        'explanation': parse_result.data
                    })

                if i < len(test_cases) - 1:
                    time.sleep(delay_seconds)

            except Exception as e:
                logger.error(f"Case {i} failed: {e}")
                responses.append({
                    'regulation_chunk_id': case.get(
                        'id', f'case_{i}'
                    ),
                    'explanation': {
                        'summary': 'Generation failed',
                        'recommendation': 'Manual review',
                        'risk_level': 'medium',
                        'key_differences': ['Error'],
                        'confidence': 'low'
                    }
                })

        # Analyze quality
        reg_texts = [
            c.get('regulation_text', '') for c in test_cases
        ]
        report = self.analyzer.analyze_batch(
            responses, reg_texts
        )

        return {
            'variant': variant_name,
            'variant_name': variant.name,
            'description': variant.description,
            'quality_report': report,
            'avg_latency_ms': (
                sum(latencies) / len(latencies)
                if latencies else 0
            ),
            'total_tested': len(test_cases),
            'tested_at': datetime.now().isoformat()
        }

    def compare_variants(
        self,
        results: List[Dict]
    ) -> Dict:
        """Compare A/B test results across variants"""
        comparison = []

        for result in results:
            report = result['quality_report']
            avg = report.get('average_scores', {})
            comparison.append({
                'variant': result['variant'],
                'name': result['variant_name'],
                'overall': avg.get('overall', 0),
                'accuracy': avg.get('accuracy', 0),
                'actionability': avg.get('actionability', 0),
                'pass_rate': report.get('pass_rate', 0),
                'avg_latency_ms': result.get(
                    'avg_latency_ms', 0
                )
            })

        # Sort by overall score
        comparison.sort(
            key=lambda x: x['overall'],
            reverse=True
        )

        return {
            'winner': comparison[0] if comparison else None,
            'ranking': comparison,
            'improvement': (
                round(
                    (comparison[0]['overall']
                     - comparison[-1]['overall']) * 100,
                    1
                )
                if len(comparison) > 1 else 0
            ),
            'compared_at': datetime.now().isoformat()
        }

    def get_best_variant(self) -> str:
        """Get name of best performing variant"""
        return 'v3_fewshot'


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Prompt Optimizer Test")
    print("=" * 60)

    optimizer = PromptOptimizer()

    print("\n1. Registered variants:")
    for name, variant in optimizer.variants.items():
        print(f"   - {name}: {variant.description}")

    print("\n2. Testing variant formatting...")
    test_case = {
        'id': 'test_001',
        'regulation_text': 'Organizations must implement MFA.',
        'policy_text': 'Users must use strong passwords.',
        'final_score': 0.28,
        'cross_encoder_score': 0.30,
        'bi_encoder_score': 0.25,
        'regulation_source': 'GDPR',
        'policy_source': 'IT Policy',
        'classification': 'gap',
        'confidence': 0.7
    }

    for name, variant in optimizer.variants.items():
        try:
            prompt = variant.user_template.format(**{
                'regulation_text': test_case['regulation_text'],
                'policy_text': test_case['policy_text'],
                'final_score': test_case['final_score'],
                'cross_encoder_score': test_case[
                    'cross_encoder_score'
                ],
                'bi_encoder_score': test_case['bi_encoder_score'],
                'regulation_source': test_case[
                    'regulation_source'
                ],
                'policy_source': test_case['policy_source'],
                'classification': test_case['classification'],
                'confidence': test_case['confidence']
            })
            print(
                f"    {name}: {len(prompt)} chars"
            )
        except KeyError as e:
            print(f"    {name}: Missing key {e}")

    print("\n3. Testing single variant with Groq...")
    print("   (Testing v2_accuracy - 1 API call)")
    result = optimizer.test_variant(
        'v2_accuracy',
        [test_case],
        delay_seconds=0
    )
    report = result['quality_report']
    print(
        f"   Pass rate: {report.get('pass_rate', 0)}%"
    )
    print(
        f"   Overall: "
        f"{report.get('average_scores', {}).get('overall', 0):.2f}"
    )
    print(f"   Latency: {result['avg_latency_ms']:.0f}ms")

    print("\n" + "=" * 60)
    print("=" * 60)

    # Add this to see what Groq actually returned
    for resp in result.get('quality_report', {}).get('individual_scores', []):
        print(f"   Issues: {resp['issues']}")