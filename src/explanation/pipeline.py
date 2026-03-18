"""Explanation Pipeline - Batch processing for Groq API"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Optional, Callable
import logging
import time
from datetime import datetime
from dataclasses import dataclass
import threading

from explanation.prompts import PromptTemplateManager
from explanation.groq_client import GroqLLMClient
from explanation.schemas import ExplanationOutput, GapExplanation
from explanation.response_parser import LLMResponseParser
from explanation.cache_manager import (
    LLMCacheManager,
    IntelligentRetryManager,
    CheckpointManager
)

from scoring.gap_classifier import GapClassification, GapClass
from scoring.unsupported_detector import UnsupportedRequirement

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    requests_per_minute: int = 30
    batch_delay_seconds: float = 2.0
    temperature: float = 0.1
    max_tokens: int = 500
    cache_ttl_days: int = 7
    enable_cache: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    skip_aligned_confidence: float = 0.85


class RateLimiter:
    """Token bucket rate limiter for Groq free tier"""

    def __init__(self, requests_per_minute: int):
        self.interval = 60.0 / requests_per_minute
        self.last_request = 0
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            now = time.time()
            elapsed = now - self.last_request
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_request = time.time()


class ExplanationPipeline:
    """
    Batch explanation pipeline for Groq API

    Features:
    - Smart filtering (skip aligned chunks)
    - Checkpoint saving every N items
    - Daily usage monitoring
    - Progress tracking
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[PipelineConfig] = None
    ):
        self.config = config or PipelineConfig()

        self.prompt_manager = PromptTemplateManager()
        self.llm_client = GroqLLMClient(
            api_key=api_key,
            model="llama3-8b-8192",
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            max_retries=self.config.max_retries
        )
        self.parser = LLMResponseParser()

        self.cache_manager = LLMCacheManager(
            default_ttl_days=self.config.cache_ttl_days
        ) if self.config.enable_cache else None

        self.retry_manager = IntelligentRetryManager(
            cache_manager=self.cache_manager or LLMCacheManager(),
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_delay
        )

        self.rate_limiter = RateLimiter(self.config.requests_per_minute)

        self.progress = {
            'total': 0, 'completed': 0,
            'failed': 0, 'cached': 0, 'skipped': 0
        }
        self._callback: Optional[Callable] = None

    def set_progress_callback(self, callback: Callable):
        self._callback = callback

    def _update_progress(self, completed: int, total: int):
        self.progress['completed'] = completed
        self.progress['total'] = total
        if self._callback:
            try:
                self._callback(completed, total)
            except Exception:
                pass

    def filter_needs_explanation(
        self,
        gaps: List[GapClassification]
    ) -> tuple:
        """
        Skip ALIGNED gaps with high confidence

        Saves ~40-60% of API calls on typical datasets
        """
        needs = []
        skipped = []

        for gap in gaps:
            if (gap.classification == GapClass.ALIGNED
                    and gap.confidence > self.config.skip_aligned_confidence):
                skipped.append(gap)
            else:
                needs.append(gap)

        logger.info(
            f"Filter: {len(needs)} need explanation, "
            f"{len(skipped)} skipped"
        )
        self.progress['skipped'] = len(skipped)
        return needs, skipped

    def generate_explanation(
        self,
        gap: GapClassification,
        prompt_version: str = "2.0.0"
    ) -> ExplanationOutput:
        """Generate explanation for single gap"""
        start = time.time()

        best_match = gap.policy_matches[0] if gap.policy_matches else {}

        variables = {
            'regulation_text': gap.regulation_text,
            'policy_text': best_match.get(
                'policy_text', 'No matching policy found'
            ),
            'final_score': gap.final_score,
            'cross_encoder_score': gap.cross_encoder_score,
            'bi_encoder_score': gap.bi_encoder_score,
            'regulation_source': gap.regulation_metadata.get(
                'source', 'Unknown'
            ),
            'policy_source': best_match.get('policy_source', 'Unknown'),
            'classification': gap.classification.value,
            'confidence': gap.confidence
        }

        prompt = self.prompt_manager.format_prompt(
            "gap_analysis", variables, prompt_version
        )

        self.rate_limiter.wait()

        result = self.retry_manager.execute(
            prompt=prompt,
            model="llama3-8b-8192",
            prompt_version=f"gap_analysis@{prompt_version}",
            llm_client=self.llm_client,
            parser=self.parser,
            system_prompt=self.prompt_manager.system_prompt,
            temperature=self.config.temperature
        )

        elapsed = (time.time() - start) * 1000

        if result['success']:
            data = result['data']
            if 'confidence' not in data:
                data['confidence'] = 'medium'

            try:
                explanation = GapExplanation(**data)
            except Exception:
                explanation = GapExplanation(
                    summary=data.get('summary', 'Analysis incomplete'),
                    recommendation=data.get(
                        'recommendation', 'Manual review'
                    ),
                    risk_level=data.get('risk_level', 'medium'),
                    key_differences=data.get(
                        'key_differences', ['Review needed']
                    ),
                    confidence='low'
                )

            return ExplanationOutput(
                explanation_type="gap",
                regulation_chunk_id=gap.regulation_chunk_id,
                policy_chunk_id=best_match.get('policy_chunk_id'),
                explanation=explanation,
                llm_model="llama3-8b-8192",
                prompt_version=f"gap_analysis@{prompt_version}",
                generation_time_ms=elapsed,
                parsing_success=True
            )
        else:
            return ExplanationOutput(
                explanation_type="gap",
                regulation_chunk_id=gap.regulation_chunk_id,
                policy_chunk_id=best_match.get('policy_chunk_id'),
                explanation=GapExplanation(
                    summary="Generation failed",
                    recommendation="Manual review required",
                    risk_level="medium",
                    key_differences=[f"Error: {result['error']}"],
                    confidence="low"
                ),
                llm_model="llama3-8b-8192",
                prompt_version=f"gap_analysis@{prompt_version}",
                generation_time_ms=elapsed,
                parsing_success=False
            )

    def generate_unsupported_explanation(
        self,
        unsupported: UnsupportedRequirement
    ) -> ExplanationOutput:
        """Generate explanation for unmatched requirement"""
        start = time.time()

        variables = {
            'regulation_text': unsupported.regulation_text,
            'regulation_source': unsupported.regulation_source,
            'page_number': unsupported.regulation_page or 'N/A',
            'severity': unsupported.severity,
            'severity_score': unsupported.severity_score
        }

        prompt = self.prompt_manager.format_prompt(
            "unsupported_analysis", variables, "1.0.0"
        )

        self.rate_limiter.wait()

        result = self.retry_manager.execute(
            prompt=prompt,
            model="llama3-8b-8192",
            prompt_version="unsupported_analysis@1.0.0",
            llm_client=self.llm_client,
            parser=self.parser,
            system_prompt=self.prompt_manager.system_prompt,
            temperature=self.config.temperature
        )

        elapsed = (time.time() - start) * 1000
        data = result.get('data', {})

        explanation = GapExplanation(
            summary=data.get('summary', 'No policy coverage'),
            recommendation=data.get('recommendation', 'Create policy'),
            risk_level=data.get('risk_level', 'high'),
            key_differences=data.get(
                'key_differences', ['No coverage found']
            ),
            confidence=data.get('confidence', 'medium'),
            regulatory_intent=f"Severity: {unsupported.severity}"
        )

        return ExplanationOutput(
            explanation_type="unsupported",
            regulation_chunk_id=unsupported.regulation_chunk_id,
            policy_chunk_id=None,
            explanation=explanation,
            llm_model="llama3-8b-8192",
            prompt_version="unsupported_analysis@1.0.0",
            generation_time_ms=elapsed,
            parsing_success=result['success']
        )

    def process_batch(
        self,
        gaps: List[GapClassification],
        unsupported: Optional[List[UnsupportedRequirement]] = None,
        prompt_version: str = "2.0.0",
        use_checkpoints: bool = True,
        checkpoint_every: int = 10
    ) -> List[ExplanationOutput]:
        """
        Process batch with smart filtering and checkpointing

        Args:
            gaps: Gap classifications from Week 4
            unsupported: Unmatched requirements
            prompt_version: Prompt template version
            use_checkpoints: Save progress periodically
            checkpoint_every: Save every N items

        Returns:
            List of ExplanationOutput
        """
        # Check daily usage before starting
        usage = self.llm_client.check_daily_usage()
        if usage['status'] == 'critical':
            logger.warning(
                f"⚠️  Near daily limit: {usage['used_today']}"
                f"/{usage['daily_limit']}"
            )

        # Smart filtering
        gaps_to_process, skipped = self.filter_needs_explanation(gaps)
        all_items = list(gaps_to_process) + list(unsupported or [])
        total = len(all_items)

        logger.info(f"Processing {total} items ({len(skipped)} skipped)")

        # Load checkpoint
        checkpoint = CheckpointManager() if use_checkpoints else None
        completed_ids = []
        results_dicts = []

        if checkpoint and checkpoint.exists():
            completed_ids, results_dicts = checkpoint.load()
            all_items = [
                item for item in all_items
                if item.regulation_chunk_id not in completed_ids
            ]
            logger.info(f"Resuming: {len(all_items)} remaining")

        results = []

        for i, item in enumerate(all_items):
            try:
                # Check daily limit
                usage = self.llm_client.check_daily_usage()
                if usage['remaining_today'] < 10:
                    logger.error("Daily API limit nearly reached. Stopping.")
                    if checkpoint:
                        checkpoint.save(completed_ids, results_dicts)
                    break

                if isinstance(item, GapClassification):
                    explanation = self.generate_explanation(
                        item, prompt_version
                    )
                else:
                    explanation = self.generate_unsupported_explanation(item)

                results.append(explanation)
                completed_ids.append(item.regulation_chunk_id)
                results_dicts.append(explanation.to_dict())

                if checkpoint and (i + 1) % checkpoint_every == 0:
                    checkpoint.save(completed_ids, results_dicts)

                self._update_progress(i + 1, total)

                if i < len(all_items) - 1:
                    time.sleep(self.config.batch_delay_seconds)

            except Exception as e:
                logger.error(f"Item {i} failed: {e}")
                self.progress['failed'] += 1
                if checkpoint:
                    checkpoint.save(completed_ids, results_dicts)
                self._update_progress(i + 1, total)

        if checkpoint and results:
            checkpoint.clear()

        logger.info(
            f"Done: {len(results)} generated, "
            f"{self.progress['skipped']} skipped, "
            f"{self.progress['failed']} failed"
        )

        return results

    def enrich_gap_report(
        self,
        gap_report: Dict,
        explanations: List[ExplanationOutput]
    ) -> Dict:
        """Add explanations to Week 4 gap report"""
        lookup = {e.regulation_chunk_id: e.to_dict() for e in explanations}

        enriched = []
        for reg in gap_report.get('regulation_analysis', []):
            if reg['regulation_chunk_id'] in lookup:
                reg['explanation'] = lookup[reg['regulation_chunk_id']]
            enriched.append(reg)

        gap_report['regulation_analysis'] = enriched
        gap_report['explanation_metadata'] = {
            'model': 'llama3-8b-8192',
            'provider': 'groq',
            'generated_at': datetime.now().isoformat(),
            'total_explanations': len(explanations),
            'skipped_aligned': self.progress['skipped'],
            'prompt_version': '2.0.0'
        }

        return gap_report

    def get_quality_metrics(
        self,
        explanations: List[ExplanationOutput]
    ) -> Dict:
        """Quality metrics for generated explanations"""
        if not explanations:
            return {}

        total = len(explanations)
        successful = sum(1 for e in explanations if e.parsing_success)
        risk = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        conf = {'high': 0, 'medium': 0, 'low': 0}
        total_time = 0

        for e in explanations:
            risk[e.explanation.risk_level.value] += 1
            conf[e.explanation.confidence.value] += 1
            total_time += e.generation_time_ms

        cache_stats = self.retry_manager.get_stats()

        return {
            'total': total,
            'skipped': self.progress['skipped'],
            'success_rate': round(successful / total * 100, 1),
            'avg_time_ms': round(total_time / total, 1),
            'risk_distribution': risk,
            'confidence_distribution': conf,
            'cache_hit_rate': round(
                cache_stats['cache_hits'] /
                max(cache_stats['total'], 1) * 100, 1
            ),
            'daily_usage': self.llm_client.check_daily_usage()
        }


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Pipeline Test")
    print("=" * 60)

    print("\n1. Smart filtering test...")
    from scoring.gap_classifier import GapClassification, GapClass

    config = PipelineConfig()

    mock_gaps = []
    cases = [
        (GapClass.ALIGNED, 0.92),
        (GapClass.PARTIAL, 0.75),
        (GapClass.GAP, 0.60),
        (GapClass.ALIGNED, 0.55),
        (GapClass.UNMATCHED, 0.80),
        (GapClass.ALIGNED, 0.88),
    ]

    for i, (cls, conf) in enumerate(cases):
        mock_gaps.append(GapClassification(
            regulation_chunk_id=f"reg_{i}",
            regulation_text=f"Test {i}",
            regulation_metadata={'source': 'Test'},
            classification=cls,
            confidence=conf,
            confidence_level="high",
            bi_encoder_score=0.6,
            cross_encoder_score=0.7,
            final_score=0.65,
            threshold_min=0.0,
            threshold_max=1.0,
            reasoning="test",
            recommended_action="test",
            policy_matches=[],
            classified_at=datetime.now().isoformat(),
            config_version="test"
        ))

    needs = [g for g in mock_gaps
             if not (g.classification == GapClass.ALIGNED
                     and g.confidence > config.skip_aligned_confidence)]
    skipped = [g for g in mock_gaps if g not in needs]

    print(f"   Total: {len(mock_gaps)}")
    print(f"   Needs explanation: {len(needs)}")
    print(f"   Skipped: {len(skipped)}")
    savings = len(skipped)/len(mock_gaps)*100
    print(f"   API call savings: {savings:.0f}%")

    print("\n2. Checkpoint test...")
    cp = CheckpointManager("outputs/checkpoints/pipeline_test.json")
    cp.save(['a', 'b', 'c'], [{'x': 1}, {'x': 2}, {'x': 3}])
    ids, res = cp.load()
    print(f"   Save/load: {'PASS' if len(ids) == 3 else 'FAIL'}")
    cp.clear()

    print("\n3. Rate limiter test...")
    limiter = RateLimiter(60)
    start = time.time()
    limiter.wait()
    limiter.wait()
    elapsed = time.time() - start
    print(f"   Two calls: {elapsed:.2f}s (expected ~1s)")

    print("\n" + "=" * 60)
    print("=" * 60)