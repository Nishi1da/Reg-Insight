""" Refined Explanation Generator - Week 6 Final"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Optional, Union
import logging
import json
import time
from datetime import datetime

from explanation.groq_client import GroqLLMClient
from explanation.response_parser import LLMResponseParser
from explanation.cache_manager import LLMCacheManager
from prompt_engineering.prompt_optimizer import PromptOptimizer
from prompt_engineering.quality_analyzer import PromptQualityAnalyzer
from prompt_engineering.structure_improver import StructureImprover
from prompt_engineering.refinement_pipeline import RefinementPipeline
from prompt_engineering.consistency_manager import ConsistencyManager
from prompt_engineering.quality_assurance import QualityAssuranceSystem

from scoring.gap_classifier import GapClassification, GapClass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RefinedExplanationGenerator:
    """
    Week 6 Final: Production-ready explanation generator

    Integrates all Week 6 improvements:
    - Best prompt variant (v3_fewshot)
    - Quality gating (reject poor explanations)
    - Auto-refinement pipeline
    - Consistency checking
    - QA validation
    - Improvement suggestions
    - Full metrics tracking

    This replaces the basic generator from Week 5
    with a higher-quality, more reliable system.
    """

    # Quality gate threshold
    QUALITY_GATE = 0.65
    # Use self-consistency for critical gaps
    CRITICAL_CONSISTENCY_THRESHOLD = 0.5

    def __init__(
        self,
        api_key: Optional[str] = None,
        use_best_prompt: bool = True,
        enable_quality_gate: bool = True,
        enable_refinement: bool = True
    ):
        self.groq_client = GroqLLMClient(api_key=api_key)
        self.parser = LLMResponseParser()
        self.cache = LLMCacheManager()

        self.prompt_optimizer = PromptOptimizer()
        self.quality_analyzer = PromptQualityAnalyzer()
        self.structure_improver = StructureImprover()
        self.refinement_pipeline = RefinementPipeline()
        self.consistency_manager = ConsistencyManager()
        self.qa_system = QualityAssuranceSystem()

        self.use_best_prompt = use_best_prompt
        self.enable_quality_gate = enable_quality_gate
        self.enable_refinement = enable_refinement

        # Use best performing prompt variant
        self.active_variant = (
            'v3_fewshot'
            if use_best_prompt
            else 'v1_baseline'
        )

        self.metrics = {
            'total_generated': 0,
            'quality_gate_passed': 0,
            'quality_gate_failed': 0,
            'refinements_applied': 0,
            'cache_hits': 0,
            'avg_quality_score': 0.0,
            'total_quality_scores': []
        }

        logger.info("RefinedExplanationGenerator ready")
        logger.info(f"  Active prompt: {self.active_variant}")
        logger.info(
            f"  Quality gate: "
            f"{'on' if enable_quality_gate else 'off'} "
            f"(threshold={self.QUALITY_GATE})"
        )
        logger.info(
            f"  Refinement: "
            f"{'on' if enable_refinement else 'off'}"
        )

    def generate(
        self,
        gap: GapClassification,
        use_consistency: bool = False
    ) -> Dict:
        """
        Generate high-quality explanation for a gap

        Args:
            gap: Gap classification from Week 4
            use_consistency: Use self-consistency check

        Returns:
            Refined, validated explanation dict
        """
        self.metrics['total_generated'] += 1

        variant = self.prompt_optimizer.variants[
            self.active_variant
        ]

        best_match = (
            gap.policy_matches[0]
            if gap.policy_matches else {}
        )

        # Format prompt
        try:
            prompt = variant.user_template.format(
                regulation_text=gap.regulation_text,
                policy_text=best_match.get(
                    'policy_text', 'No matching policy found'
                ),
                final_score=gap.final_score,
                cross_encoder_score=gap.cross_encoder_score,
                bi_encoder_score=gap.bi_encoder_score,
                regulation_source=gap.regulation_metadata.get(
                    'source', 'Unknown'
                ),
                policy_source=best_match.get(
                    'policy_source', 'Unknown'
                ),
                classification=gap.classification.value,
                confidence=gap.confidence
            )
        except KeyError as e:
            logger.error(f"Prompt formatting failed: {e}")
            return self._fallback_explanation(gap)

        # Check cache first
        cached = self.cache.get(
            prompt, "llama-3.1-8b-instant",
            f"refined_{self.active_variant}"
        )
        if cached:
            self.metrics['cache_hits'] += 1
            return cached

        # Use self-consistency for ambiguous gaps
        should_use_consistency = (
            use_consistency
            or (gap.confidence < self.CRITICAL_CONSISTENCY_THRESHOLD
                and gap.classification == GapClass.PARTIAL)
        )

        if should_use_consistency:
            result = self.consistency_manager.self_consistency_check(
                prompt=prompt,
                system_prompt=variant.system_prompt,
                num_generations=3,
                temperature=0.1,
                delay_seconds=2.0
            )
            explanation_data = (
                result.get('best_response', {})
                if result.get('success')
                else {}
            )
        else:
            response = self.groq_client.generate(
                prompt=prompt,
                system_prompt=variant.system_prompt,
                temperature=0.1
            )
            parse_result = self.parser.parse(response.text)
            explanation_data = (
                parse_result.data
                if parse_result.success
                else {}
            )

        if not explanation_data:
            return self._fallback_explanation(gap)

        # Apply refinement pipeline
        if self.enable_refinement:
            refined = self.refinement_pipeline.refine(
                {'explanation': explanation_data},
                regulation_text=gap.regulation_text,
                policy_text=best_match.get('policy_text', '')
            )
            explanation_data = refined.refined
            if refined.fixes_applied:
                self.metrics['refinements_applied'] += len(
                    refined.fixes_applied
                )

        # Quality gate
        quality = self.quality_analyzer.analyze(
            {'explanation': explanation_data},
            gap.regulation_text,
            best_match.get('policy_text', '')
        )

        self.metrics['total_quality_scores'].append(
            quality.overall_score
        )
        self.metrics['avg_quality_score'] = (
            sum(self.metrics['total_quality_scores'])
            / len(self.metrics['total_quality_scores'])
        )

        if (self.enable_quality_gate
                and not quality.passed
                and quality.overall_score < self.QUALITY_GATE):
            self.metrics['quality_gate_failed'] += 1
            logger.warning(
                f"Quality gate failed for "
                f"{gap.regulation_chunk_id}: "
                f"{quality.overall_score:.2f}"
            )
            # Try once more with different temperature
            retry_response = self.groq_client.generate(
                prompt=prompt,
                system_prompt=variant.system_prompt,
                temperature=0.3
            )
            retry_result = self.parser.parse(
                retry_response.text
            )
            if retry_result.success:
                retry_quality = self.quality_analyzer.analyze(
                    {'explanation': retry_result.data}
                )
                if (retry_quality.overall_score
                        > quality.overall_score):
                    explanation_data = retry_result.data
                    quality = retry_quality
        else:
            self.metrics['quality_gate_passed'] += 1

        output = {
            'regulation_chunk_id': gap.regulation_chunk_id,
            'explanation_type': 'gap',
            'explanation': explanation_data,
            'quality_score': quality.overall_score,
            'quality_passed': quality.passed,
            'prompt_version': self.active_variant,
            'generated_at': datetime.now().isoformat(),
            'issues': quality.issues if quality.issues else []
        }

        # Cache the result
        self.cache.set(
            prompt,
            "llama-3.1-8b-instant",
            f"refined_{self.active_variant}",
            output
        )

        return output

    def generate_batch(
        self,
        gaps: List[GapClassification],
        delay_seconds: float = 2.0,
        progress_callback=None
    ) -> List[Dict]:
        """Generate refined explanations for batch"""
        results = []
        total = len(gaps)

        # Check daily usage
        usage = self.groq_client.check_daily_usage()
        if usage['status'] == 'critical':
            logger.error("Near daily API limit. Aborting.")
            return []

        logger.info(
            f"Generating {total} refined explanations..."
        )

        for i, gap in enumerate(gaps):
            try:
                result = self.generate(gap)
                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, total)

                if i < total - 1:
                    time.sleep(delay_seconds)

            except Exception as e:
                logger.error(
                    f"Gap {gap.regulation_chunk_id} failed: {e}"
                )
                results.append(
                    self._fallback_explanation(gap)
                )

        return results

    def _fallback_explanation(
        self,
        gap: GapClassification
    ) -> Dict:
        """Fallback when generation fails"""
        return {
            'regulation_chunk_id': gap.regulation_chunk_id,
            'explanation_type': 'gap',
            'explanation': {
                'summary': (
                    'Automated analysis incomplete - '
                    'manual review required.'
                ),
                'recommendation': (
                    'Review regulation against company '
                    'policies manually.'
                ),
                'risk_level': 'medium',
                'key_differences': [
                    'Manual review needed'
                ],
                'confidence': 'low'
            },
            'quality_score': 0.0,
            'quality_passed': False,
            'prompt_version': 'fallback',
            'generated_at': datetime.now().isoformat(),
            'issues': ['generation_failed']
        }

    def run_qa_validation(
        self,
        explanations: List[Dict]
    ) -> Dict:
        """Run QA validation on generated explanations"""
        checks = self.qa_system.run_automated_checks(
            explanations
        )
        flagged = self.qa_system.flag_low_confidence(
            explanations,
            confidence_threshold=self.QUALITY_GATE
        )

        return {
            'automated_checks': checks,
            'flagged_for_review': flagged,
            'total_flagged': len(flagged),
            'overall_health': (
                'good' if checks.get('checks_passed')
                else 'needs_improvement'
            )
        }

    def get_improvement_suggestions(
        self,
        explanations: List[Dict]
    ) -> List[str]:
        """Get suggestions to improve prompt quality"""
        if not explanations:
            return []

        reg_texts = [
            e.get('regulation_text', '')
            for e in explanations
        ]
        report = self.quality_analyzer.analyze_batch(
            explanations, reg_texts
        )
        return self.quality_analyzer.get_improvement_suggestions(
            report
        )

    def get_metrics(self) -> Dict:
        """Get generation metrics"""
        total = self.metrics['total_generated']
        return {
            'total_generated': total,
            'cache_hit_rate': round(
                self.metrics['cache_hits'] / total * 100
                if total > 0 else 0, 1
            ),
            'quality_gate_pass_rate': round(
                self.metrics['quality_gate_passed'] / total * 100
                if total > 0 else 0, 1
            ),
            'avg_quality_score': round(
                self.metrics['avg_quality_score'], 3
            ),
            'avg_refinements_per_explanation': round(
                self.metrics['refinements_applied'] / total
                if total > 0 else 0, 1
            ),
            'active_prompt': self.active_variant,
            'groq_usage': self.groq_client.check_daily_usage()
        }

    def get_final_prompt(self) -> Dict:
        """Get the finalized prompt for documentation"""
        variant = self.prompt_optimizer.variants[
            self.active_variant
        ]
        return {
            'version': self.active_variant,
            'name': variant.name,
            'description': variant.description,
            'system_prompt': variant.system_prompt,
            'user_template': variant.user_template,
            'rationale': (
                "Selected based on A/B testing results. "
                "Few-shot examples significantly improve "
                "consistency and accuracy of JSON output. "
                "Constraints reduce hallucinations. "
                "Chain-of-thought not needed as Llama-3.1 "
                "performs well with examples alone."
            ),
            'finalized_at': datetime.now().isoformat()
        }


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Refined Explanation Generator Test")
    print("=" * 60)

    print("\n1. Initializing refined generator...")
    try:
        gen = RefinedExplanationGenerator(
            use_best_prompt=True,
            enable_quality_gate=True,
            enable_refinement=True
        )
        print("    Generator ready")
        print(f"   Active prompt: {gen.active_variant}")
        print(f"   Quality gate: {gen.QUALITY_GATE}")
    except Exception as e:
        print(f"    Init failed: {e}")
        exit()

    print("\n2. Testing single generation...")
    mock_gap = GapClassification(
        regulation_chunk_id="test_001",
        regulation_text=(
            "Organizations must implement multi-factor "
            "authentication for all privileged accounts."
        ),
        regulation_metadata={'source': 'ISO 27001'},
        classification=GapClass.GAP,
        confidence=0.7,
        confidence_level="medium",
        bi_encoder_score=0.28,
        cross_encoder_score=0.32,
        final_score=0.30,
        threshold_min=0.0,
        threshold_max=0.39,
        reasoning="Low match score",
        recommended_action="Create MFA policy",
        policy_matches=[{
            'policy_text': (
                'Users must use strong passwords '
                'of 12 characters minimum.'
            ),
            'policy_source': 'IT Security Policy',
            'policy_chunk_id': 'pol_001'
        }],
        classified_at=datetime.now().isoformat(),
        config_version="v1"
    )

    result = gen.generate(mock_gap)
    exp = result.get('explanation', {})
    print(f"    Generated successfully")
    print(
        f"   Summary: {str(exp.get('summary', ''))[:70]}..."
    )
    risk = exp.get('risk_level', 'N/A')
    if hasattr(risk, 'value'):
        risk = risk.value
        print(f"   Risk: {risk}")
    print(
        f"   Quality score: {result.get('quality_score', 0):.2f}"
    )
    print(
        f"   Quality passed: {result.get('quality_passed')}"
    )
    print(f"   Issues: {result.get('issues', [])}")

    print("\n3. QA validation test...")
    qa_result = gen.run_qa_validation([result])
    print(
        f"   Health: {qa_result['overall_health']}"
    )
    print(
        f"   Flagged: {qa_result['total_flagged']} items"
    )

    print("\n4. Final prompt documentation...")
    final_prompt = gen.get_final_prompt()
    print(f"   Version: {final_prompt['version']}")
    print(f"   Name: {final_prompt['name']}")
    print(f"   Rationale: {final_prompt['rationale'][:80]}...")

    print("\n5. Metrics...")
    metrics = gen.get_metrics()
    print(f"   Total generated: {metrics['total_generated']}")
    print(
        f"   Quality gate pass rate: "
        f"{metrics['quality_gate_pass_rate']}%"
    )
    print(
        f"   Avg quality score: "
        f"{metrics['avg_quality_score']:.2f}"
    )
    print(
        f"   Groq usage: "
        f"{metrics['groq_usage']['used_today']}/"
        f"{metrics['groq_usage']['daily_limit']}"
    )

    print("\n" + "=" * 60)
    print(" Week 6 finished!")
    print("=" * 60)
    print("\nWeek 6 Deliverables:")
    print("   5-dimension quality rubric")
    print("   4 prompt variants with A/B testing")
    print("   Few-shot prompting for accuracy")
    print("   Structure validation + auto-fix")
    print("   Self-consistency checking")
    print("   Temperature tuning")
    print("   A/B test result logging (SQLite)")
    print("   Text cleaning + terminology standardization")
    print("   Explanation ranking")
    print("   Golden set QA (30 examples)")
    print("   Automated quality checks")
    print("   Quality dashboard")
    print("   Low confidence flagging")
    print("   Quality gating (reject poor outputs)")
    print("   Improvement suggestions")
    print("   Finalized prompt v3_fewshot")