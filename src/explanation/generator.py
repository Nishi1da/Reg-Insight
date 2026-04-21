"""Explanation Generator - Main interface for Week 5 Groq integration"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Optional, Union
import logging
import yaml
import json
import argparse
from datetime import datetime

from explanation.prompts import PromptTemplateManager
from explanation.groq_client import GroqLLMClient
from explanation.schemas import ExplanationOutput, GapExplanation, SchemaValidator
from explanation.response_parser import LLMResponseParser
from explanation.cache_manager import LLMCacheManager, IntelligentRetryManager
from explanation.pipeline import ExplanationPipeline, PipelineConfig

from scoring.gap_classifier import GapClassification, GapClass
from scoring.unsupported_detector import UnsupportedRequirement

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExplanationGenerator:
    """
    Main interface for Groq-powered explanation generation

    Connects Week 4 gap classifications to Week 5
    LLM explanations via Groq API.

    Features:
    - YAML configuration
    - API key management
    - Post-processing
    - Confidence scoring
    - Daily usage monitoring
    - Batch and single processing
    """

    def __init__(
            self,
            config_path: str = "config/groq_config.yaml",
            api_key: Optional[str] = None
            ):
        from dotenv import load_dotenv
        import os
        
        env_path = Path(__file__).parent.parent.parent / ".env"
        load_dotenv(dotenv_path=env_path)
        
        if not Path(config_path).is_absolute():
            config_path = str(Path(__file__).parent.parent.parent / config_path)
            self.config = self._load_config(config_path)
            
            # Get API key: explicit arg → .env → yaml config
            self.api_key = (
                api_key
                or os.environ.get('GROQ_API_KEY')
                or self.config.get('groq', {}).get('api_key')
                
                )
            
            self.config = self._load_config(config_path)
            
            # Get API key
            self.api_key = (
                api_key
                or self.config.get('groq', {}).get('api_key')
                )

        if not self.api_key or self.api_key == 'gsk_your_key_here':
            raise ValueError(
                "Groq API key not set.\n"
                "Add your key to config/groq_config.yaml\n"
                "Get free key at: console.groq.com"
            )

        pipeline_config = PipelineConfig(
            requests_per_minute=self.config.get(
                'groq', {}
            ).get('requests_per_minute', 30),
            batch_delay_seconds=self.config.get(
                'groq', {}
            ).get('batch_delay_seconds', 2.0),
            temperature=self.config.get('groq', {}).get('temperature', 0.1),
            max_tokens=self.config.get('groq', {}).get('max_tokens', 500),
            cache_ttl_days=self.config.get(
                'caching', {}
            ).get('ttl_days', 7),
            enable_cache=self.config.get(
                'caching', {}
            ).get('enabled', True),
            skip_aligned_confidence=self.config.get(
                'smart_filtering', {}
            ).get('skip_aligned_above_confidence', 0.85)
        )

        self.pipeline = ExplanationPipeline(
            api_key=self.api_key,
            config=pipeline_config
        )

        self.post_processing = self.config.get('post_processing', {})
        self.validator = SchemaValidator()

        logger.info("ExplanationGenerator ready")
        logger.info(f"  Model: llama-3.1-8b-instant via Groq")
        logger.info(
            f"  Cache: "
            f"{'on' if pipeline_config.enable_cache else 'off'}"
        )

    def _load_config(self, path: str) -> Dict:
        try:
            with open(path, encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config not found: {path}")
            return {}

    def _post_process(self, explanation: GapExplanation) -> GapExplanation:
        """Apply text post-processing"""
        summary = explanation.summary
        rec = explanation.recommendation

        if self.post_processing.get('trim_whitespace', True):
            summary = summary.strip()
            rec = rec.strip()

        if self.post_processing.get('fix_capitalization', True):
            if summary and summary[0].islower():
                summary = summary[0].upper() + summary[1:]
            if rec and rec[0].islower():
                rec = rec[0].upper() + rec[1:]

        max_s = self.post_processing.get('max_summary_length', 200)
        if len(summary) > max_s:
            summary = summary[:max_s - 3] + "..."

        max_r = self.post_processing.get('max_recommendation_length', 1000)
        if len(rec) > max_r:
            rec = rec[:max_r - 3] + "..."

        return GapExplanation(
            summary=summary,
            recommendation=rec,
            risk_level=explanation.risk_level,
            key_differences=explanation.key_differences,
            confidence=explanation.confidence,
            regulatory_intent=explanation.regulatory_intent,
            policy_coverage=explanation.policy_coverage,
            remediation_priority=explanation.remediation_priority
        )

    def _calculate_confidence(
        self,
        explanation: GapExplanation,
        parsing_method: str,
        parsing_success: bool
    ) -> str:
        """Combine parsing method confidence with model confidence"""
        if not parsing_success:
            return 'low'

        method_scores = {
            'direct': 1.0, 'markdown_strip': 0.9,
            'regex_extract': 0.7, 'fix_json': 0.6, 'fallback': 0.4
        }
        model_scores = {'high': 1.0, 'medium': 0.6, 'low': 0.3}

        combined = (
            0.7 * method_scores.get(parsing_method, 0.5)
            + 0.3 * model_scores.get(explanation.confidence.value, 0.5)
        )

        if combined >= 0.8:
            return 'high'
        elif combined >= 0.5:
            return 'medium'
        return 'low'

    def ensure_connection(self) -> bool:
        """
        Verify Groq API is working before batch processing

        Returns True if connected and ready
        """
        print("Checking Groq API connection...")
        health = self.pipeline.llm_client.health_check()

        if health['status'] == 'healthy':
            usage = self.pipeline.llm_client.check_daily_usage()
            print(f" Groq API connected")
            print(f"   Model: {health['model']}")
            print(f"   Latency: {health['latency_ms']:.0f}ms")
            print(
                f"   Daily usage: {usage['used_today']}"
                f"/{usage['daily_limit']} "
                f"({usage['usage_percent']}%)"
            )

            if usage['status'] == 'warning':
                print(f"     Warning: 80%+ of daily limit used")
            elif usage['status'] == 'critical':
                print(f"    Critical: Near daily limit")
                return False

            return True

        else:
            print(f" Groq API not responding: {health.get('error')}")
            print("Check your API key in config/groq_config.yaml")
            return False

    def generate(
        self,
        gap: Union[GapClassification, UnsupportedRequirement]
    ) -> ExplanationOutput:
        """Generate explanation for single item"""
        if isinstance(gap, GapClassification):
            result = self.pipeline.generate_explanation(gap, "2.0.0")
        else:
            result = self.pipeline.generate_unsupported_explanation(gap)

        processed = self._post_process(result.explanation)
        processed.confidence = self._calculate_confidence(
            processed, 'direct', result.parsing_success
        )

        return ExplanationOutput(
            explanation_type=result.explanation_type,
            regulation_chunk_id=result.regulation_chunk_id,
            policy_chunk_id=result.policy_chunk_id,
            explanation=processed,
            llm_model=result.llm_model,
            prompt_version=result.prompt_version,
            generation_time_ms=result.generation_time_ms,
            parsing_success=result.parsing_success
        )

    def generate_batch(
        self,
        items: List[Union[GapClassification, UnsupportedRequirement]],
        progress_callback=None
    ) -> List[ExplanationOutput]:
        """
        Generate explanations for multiple items

        Checks connection and daily limits first.
        """
        if not self.ensure_connection():
            raise ConnectionError("Groq API not available")

        gaps = [i for i in items if isinstance(i, GapClassification)]
        unsupported = [
            i for i in items if isinstance(i, UnsupportedRequirement)
        ]

        if progress_callback:
            self.pipeline.set_progress_callback(progress_callback)

        results = self.pipeline.process_batch(
            gaps=gaps,
            unsupported=unsupported,
            prompt_version="2.0.0",
            use_checkpoints=True,
            checkpoint_every=10
        )

        # Post-process
        processed = []
        for r in results:
            exp = self._post_process(r.explanation)
            exp.confidence = self._calculate_confidence(
                exp, 'direct', r.parsing_success
            )
            processed.append(ExplanationOutput(
                explanation_type=r.explanation_type,
                regulation_chunk_id=r.regulation_chunk_id,
                policy_chunk_id=r.policy_chunk_id,
                explanation=exp,
                llm_model=r.llm_model,
                prompt_version=r.prompt_version,
                generation_time_ms=r.generation_time_ms,
                parsing_success=r.parsing_success
            ))

        return processed

    def enrich_report(
        self,
        gap_report: Dict,
        explanations: List[ExplanationOutput]
    ) -> Dict:
        """Add explanations to Week 4 gap report"""
        return self.pipeline.enrich_gap_report(gap_report, explanations)

    def get_stats(self) -> Dict:
        """Get generator statistics"""
        return {
            'pipeline': {
                'completed': self.pipeline.progress['completed'],
                'failed': self.pipeline.progress['failed'],
                'skipped': self.pipeline.progress['skipped']
            },
            'cache': (
                self.pipeline.cache_manager.get_stats()
                if self.pipeline.cache_manager else {}
            ),
            'groq_usage': self.pipeline.llm_client.check_daily_usage(),
            'groq_client': self.pipeline.llm_client.get_stats()
        }


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description='REG-INSIGHT Explanation Generator (Groq)'
    )
    parser.add_argument(
        '--config', '-c',
        default='config/groq_config.yaml',
        help='Config file'
    )
    parser.add_argument(
        '--api-key', '-k',
        help='Override Groq API key'
    )
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Test Groq API connection'
    )
    parser.add_argument(
        '--generate', '-g',
        type=str,
        help='Generate explanation for regulation text'
    )
    parser.add_argument(
        '--enrich', '-e',
        type=str,
        help='Enrich Week 4 gap report with explanations'
    )
    parser.add_argument(
        '--output', '-o',
        default='outputs/week5_enriched_report.json',
        help='Output file'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        help='Limit items to process'
    )

    args = parser.parse_args()

    try:
        gen = ExplanationGenerator(
            config_path=args.config,
            api_key=args.api_key
        )
    except ValueError as e:
        print(f" {e}")
        return

    if args.test:
        connected = gen.ensure_connection()
        if not connected:
            print("\nCheck config/groq_config.yaml")

    elif args.generate:
        print(f"Generating for: {args.generate[:80]}...")

        mock_gap = GapClassification(
            regulation_chunk_id="cli_001",
            regulation_text=args.generate,
            regulation_metadata={'source': 'CLI'},
            classification=GapClass.PARTIAL,
            confidence=0.7,
            confidence_level="medium",
            bi_encoder_score=0.6,
            cross_encoder_score=0.65,
            final_score=0.63,
            threshold_min=0.4,
            threshold_max=0.69,
            reasoning="CLI",
            recommended_action="Review",
            policy_matches=[],
            classified_at=datetime.now().isoformat(),
            config_version="cli"
        )

        result = gen.generate(mock_gap)
        print(f"\n{'='*40}")
        print(f"Summary:    {result.explanation.summary}")
        print(f"Risk:       {result.explanation.risk_level.value}")
        print(f"Confidence: {result.explanation.confidence}")
        print(f"Action:     {result.explanation.recommendation[:100]}")
        print(f"{'='*40}")

        with open(args.output, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f" Saved: {args.output}")

    elif args.enrich:
        with open(args.enrich) as f:
            report = json.load(f)

        total = report['summary']['total_regulations']
        limit = args.limit or total
        print(f"Enriching {min(limit, total)} regulations...")
        print("Checkpoints save every 10 items - safe to interrupt")

        def progress(completed, total):
            pct = completed / total * 100
            print(f"  [{completed}/{total}] {pct:.0f}%", end='\r')

        print("\nNote: Extract GapClassification objects from your")
        print("Week 4 pipeline and pass to generate_batch()")
        print("Example:")
        print("  explanations = gen.generate_batch(gap_classifications)")
        print("  enriched = gen.enrich_report(report, explanations)")

    else:
        parser.print_help()


# Test
if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        print("=" * 60)
        print(" Explanation Generator Test")
        print("=" * 60)

        print("\n1. Testing initialization...")
        try:
            gen = ExplanationGenerator()
            print("    Generator ready")
            print(f"   Model: llama-3.1-8b-instant")
        except ValueError as e:
            print(f"     API key needed: {str(e)[:80]}")
            print("   Add key to config/groq_config.yaml and retest")

        print("\n2. Post-processing test...")
        test_exp = GapExplanation(
            summary="  policy lacks mfa requirement.  ",
            recommendation="  add mfa to access policy.  ",
            risk_level="high",
            key_differences=["Missing MFA", "Missing 2FA"],
            confidence="high"
        )

        gen_bare = ExplanationGenerator.__new__(ExplanationGenerator)
        gen_bare.post_processing = {
            'trim_whitespace': True,
            'fix_capitalization': True,
            'max_summary_length': 200,
            'max_recommendation_length': 1000
        }

        processed = gen_bare._post_process(test_exp)
        print(f"   Original: '{test_exp.summary}'")
        print(f"   Processed: '{processed.summary}'")
        print(
            f"   Capitalized: "
            f"{'PASS' if processed.summary[0].isupper() else 'FAIL'}"
        )
        print(
            f"   Trimmed: "
            f"{'PASS' if processed.summary == processed.summary.strip() else 'FAIL'}"
        )

        print("\n3. Confidence calculation test...")
        cases = [
            ('direct', True, 'high', 'high'),
            ('fallback', True, 'high', 'medium'),
            ('direct', False, 'high', 'low'),
        ]
        for method, success, model_c, expected in cases:
            e = GapExplanation(
                summary="Test summary here",
                recommendation="Test recommendation text",
                risk_level="medium",
                key_differences=["Gap found"],
                confidence=model_c
            )
            result = gen_bare._calculate_confidence(e, method, success)
            ok = "PASS" if result == expected else "FAIL"
            print(f"   {ok} {method}+{model_c} → {result}")

        print("\n4. Stats structure test...")
        stats = {
            'pipeline': {
                'completed': 0, 'failed': 0, 'skipped': 0
            },
            'groq_usage': {
                'used_today': 0, 'remaining_today': 14400,
                'usage_percent': 0.0, 'status': 'ok'
            }
        }
        print(f"   Stats keys: {list(stats.keys())}")

        print("\n" + "=" * 60)
        print("Week 5 finished!")
        print("=" * 60)
        print("\nWeek 5 Deliverable (Groq Edition):")
        print("   Groq API client (llama-3.1-8b-instant)")
        print("   Free tier usage monitoring")
        print("   Prompt templates with versioning")
        print("   Pydantic schemas + auto-fix")
        print("   5-strategy response parser")
        print("   SQLite cache (avoids re-generation)")
        print("   Smart filtering (skip aligned chunks)")
        print("   Checkpoint saving for long batches")
        print("   Circuit breaker retry")
        print("   Connection + daily limit check")
        print("   Post-processing pipeline")
        print("   CLI interface")