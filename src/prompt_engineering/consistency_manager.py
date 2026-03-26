""" Consistency Manager - Temperature tuning and self-consistency"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List, Optional, Tuple
import json
import logging
import time
import sqlite3
from datetime import datetime
from collections import Counter

from explanation.groq_client import GroqLLMClient
from explanation.response_parser import LLMResponseParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConsistencyManager:
    """
    Manages consistency of LLM outputs

    Features:
    - Self-consistency check (generate 3, pick best)
    - Temperature tuning
    - A/B test result logging
    - Inter-generation agreement metrics
    """

    def __init__(
        self,
        db_path: str = "data/ab_test_results.db"
    ):
        self.groq_client = GroqLLMClient()
        self.parser = LLMResponseParser()
        self.db_path = db_path
        self._init_db()

        self.stats = {
            'total_consistency_checks': 0,
            'high_agreement': 0,
            'low_agreement': 0
        }

    def _init_db(self):
        """Initialize A/B test results database"""
        import os
        os.makedirs(
            str(Path(self.db_path).parent),
            exist_ok=True
        )
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ab_results (
                    id TEXT PRIMARY KEY,
                    variant_name TEXT,
                    test_case_id TEXT,
                    overall_score REAL,
                    accuracy_score REAL,
                    actionability_score REAL,
                    pass_rate REAL,
                    avg_latency_ms REAL,
                    tested_at TEXT,
                    notes TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS consistency_logs (
                    id TEXT PRIMARY KEY,
                    prompt_hash TEXT,
                    temperature REAL,
                    num_generations INTEGER,
                    agreement_score REAL,
                    winning_risk_level TEXT,
                    tested_at TEXT
                )
            """)
            conn.commit()

    def self_consistency_check(
        self,
        prompt: str,
        system_prompt: str,
        num_generations: int = 3,
        temperature: float = 0.1,
        delay_seconds: float = 2.0
    ) -> Dict:
        """
        Generate multiple responses and pick most consistent

        Args:
            prompt: The prompt to test
            system_prompt: System instructions
            num_generations: How many times to generate
            temperature: LLM temperature
            delay_seconds: Delay between calls

        Returns:
            Best response with agreement metrics
        """
        self.stats['total_consistency_checks'] += 1
        generations = []

        logger.info(
            f"Self-consistency: generating "
            f"{num_generations} responses..."
        )

        for i in range(num_generations):
            try:
                response = self.groq_client.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature
                )
                result = self.parser.parse(response.text)
                if result.success:
                    generations.append(result.data)

                if i < num_generations - 1:
                    time.sleep(delay_seconds)

            except Exception as e:
                logger.error(f"Generation {i} failed: {e}")

        if not generations:
            return {
                'success': False,
                'error': 'All generations failed'
            }

        # Find most consistent response
        best, agreement = self._find_most_consistent(
            generations
        )

        if agreement >= 0.7:
            self.stats['high_agreement'] += 1
        else:
            self.stats['low_agreement'] += 1

        # Log to database
        import hashlib
        prompt_hash = hashlib.md5(
            prompt.encode()
        ).hexdigest()[:16]
        
        # Get normalized risk level for logging
        log_risk = best.get('risk_level', 'unknown')
        if hasattr(log_risk, 'value'):
            log_risk = log_risk.value
        log_risk = str(log_risk).lower()
        
        self._log_consistency(
            prompt_hash, temperature,
            num_generations, agreement,
            log_risk
        )

        # Normalize risk levels for display
        normalized_risk_levels = []
        for g in generations:
            risk = g.get('risk_level', 'unknown')
            if hasattr(risk, 'value'):
                risk = risk.value
            normalized_risk_levels.append(str(risk).lower())

        return {
            'success': True,
            'best_response': best,
            'agreement_score': agreement,
            'num_generations': len(generations),
            'all_risk_levels': normalized_risk_levels,
            'high_confidence': agreement >= 0.7
        }

    def _find_most_consistent(
        self,
        generations: List[Dict]
    ) -> Tuple[Dict, float]:
        """Find most consistent response across generations"""
        if len(generations) == 1:
            best = dict(generations[0])
            for norm_key in ['risk_level', 'confidence']:
                norm_val = best.get(norm_key, '')
                if hasattr(norm_val, 'value'):
                    best[norm_key] = norm_val.value
            return best, 1.0

        # Count risk level agreement - EXTRACT ENUM FIRST
        risk_levels = []
        for g in generations:
            risk = g.get('risk_level', 'unknown')
            if hasattr(risk, 'value'):
                risk = risk.value
            risk_levels.append(str(risk).lower())

        risk_counter = Counter(risk_levels)
        most_common_risk, count = risk_counter.most_common(1)[0]
        risk_agreement = count / len(generations)

        # Find generation matching most common risk - EXTRACT ENUM FIRST
        best = None
        for gen in generations:
            risk = gen.get('risk_level', '')
            if hasattr(risk, 'value'):
                risk = risk.value
            risk = str(risk).lower()
            if risk == most_common_risk:
                best = dict(gen)
                break

        if not best:
            best = dict(generations[0])

        # Calculate overall agreement score - EXTRACT ENUM FIRST
        conf_levels = []
        for g in generations:
            conf = g.get('confidence', 'medium')
            if hasattr(conf, 'value'):
                conf = conf.value
            conf_levels.append(str(conf).lower())

        conf_counter = Counter(conf_levels)
        conf_agreement = (
            conf_counter.most_common(1)[0][1] / len(generations)
        )

        agreement_score = (
            risk_agreement * 0.6 + conf_agreement * 0.4
        )
        
        # Normalize best response before returning
        for norm_key in ['risk_level', 'confidence']:
            norm_val = best.get(norm_key, '')
            if hasattr(norm_val, 'value'):
                best[norm_key] = norm_val.value

        return best, round(agreement_score, 3)

    def tune_temperature(
        self,
        prompt: str,
        system_prompt: str,
        temperatures: List[float] = [0.0, 0.1, 0.3, 0.5],
        delay_seconds: float = 2.0
    ) -> Dict:
        """
        Test different temperatures and find optimal

        Lower temperature = more consistent but less creative
        Higher temperature = more varied but less consistent
        """
        results = {}

        logger.info(
            f"Testing {len(temperatures)} temperatures..."
        )

        for temp in temperatures:
            responses = []
            latencies = []

            # Generate 2 responses per temperature
            for _ in range(2):
                try:
                    start = time.time()
                    response = self.groq_client.generate(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temp
                    )
                    latency = (time.time() - start) * 1000
                    latencies.append(latency)

                    result = self.parser.parse(response.text)
                    if result.success:
                        responses.append(result.data)

                    time.sleep(delay_seconds)

                except Exception as e:
                    logger.error(f"Temp {temp} failed: {e}")

            if len(responses) >= 2:
                _, agreement = self._find_most_consistent(
                    responses
                )
            else:
                agreement = 0.0

            results[str(temp)] = {
                'temperature': temp,
                'agreement_score': agreement,
                'avg_latency_ms': (
                    sum(latencies) / len(latencies)
                    if latencies else 0
                ),
                'parse_success_rate': (
                    len(responses) / 2 * 100
                )
            }

        # Find optimal temperature
        best_temp = max(
            results.items(),
            key=lambda x: x[1]['agreement_score']
        )

        return {
            'results': results,
            'optimal_temperature': best_temp[1]['temperature'],
            'optimal_agreement': best_temp[1]['agreement_score'],
            'recommendation': (
                f"Use temperature={best_temp[1]['temperature']} "
                f"for best consistency"
            )
        }

    def log_ab_result(
        self,
        variant_name: str,
        test_case_id: str,
        scores: Dict,
        notes: str = ""
    ):
        """Log A/B test result to database"""
        import hashlib
        result_id = hashlib.md5(
            f"{variant_name}{test_case_id}"
            f"{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO ab_results VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result_id,
                variant_name,
                test_case_id,
                scores.get('overall', 0),
                scores.get('accuracy', 0),
                scores.get('actionability', 0),
                scores.get('pass_rate', 0),
                scores.get('avg_latency_ms', 0),
                datetime.now().isoformat(),
                notes
            ))
            conn.commit()

    def get_ab_results(self) -> List[Dict]:
        """Get all A/B test results from database"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT variant_name,
                       AVG(overall_score) as avg_overall,
                       AVG(accuracy_score) as avg_accuracy,
                       AVG(pass_rate) as avg_pass_rate,
                       COUNT(*) as test_count
                FROM ab_results
                GROUP BY variant_name
                ORDER BY avg_overall DESC
            """).fetchall()

        return [
            {
                'variant': row[0],
                'avg_overall': round(row[1], 3),
                'avg_accuracy': round(row[2], 3),
                'avg_pass_rate': round(row[3], 1),
                'test_count': row[4]
            }
            for row in rows
        ]

    def _log_consistency(
        self,
        prompt_hash: str,
        temperature: float,
        num_generations: int,
        agreement_score: float,
        winning_risk: str
    ):
        """Log consistency check to database"""
        import hashlib
        log_id = hashlib.md5(
            f"{prompt_hash}{datetime.now().isoformat()}"
            .encode()
        ).hexdigest()[:16]

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO consistency_logs
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                log_id, prompt_hash, temperature,
                num_generations, agreement_score,
                winning_risk, datetime.now().isoformat()
            ))
            conn.commit()

    def get_stats(self) -> Dict:
        """Get consistency statistics"""
        total = self.stats['total_consistency_checks']
        return {
            **self.stats,
            'high_agreement_rate': round(
                self.stats['high_agreement'] / total * 100
                if total > 0 else 0, 1
            )
        }


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Consistency Manager Test")
    print("=" * 60)

    manager = ConsistencyManager()

    print("\n1. Self-consistency check (3 generations)...")
    test_prompt = """Analyze this compliance gap:

REGULATION: Organizations must implement data backup procedures
POLICY: We maintain system uptime through redundancy
SCORE: 0.35/1.0

Respond with JSON only:
{
    "summary": "gap assessment",
    "recommendation": "specific action",
    "risk_level": "low|medium|high|critical",
    "key_differences": ["gap 1", "gap 2"],
    "confidence": "high|medium|low"
}"""

    result = manager.self_consistency_check(
        prompt=test_prompt,
        system_prompt=(
            "You are a compliance analyst. "
            "Respond with JSON only."
        ),
        num_generations=3,
        temperature=0.1,
        delay_seconds=2.0
    )

    if result['success']:
        print(f"    Agreement score: "
              f"{result['agreement_score']:.2f}")
        print(f"   Risk levels: {result['all_risk_levels']}")
        print(
            f"   High confidence: {result['high_confidence']}"
        )
        print(
            f"   Best response risk: "
            f"{result['best_response'].get('risk_level')}"
        )
    else:
        print(f"    Failed: {result.get('error')}")

    print("\n2. Logging A/B test result...")
    manager.log_ab_result(
        variant_name='v3_fewshot',
        test_case_id='test_001',
        scores={
            'overall': 0.82,
            'accuracy': 0.85,
            'actionability': 0.78,
            'pass_rate': 80.0,
            'avg_latency_ms': 750
        },
        notes='Few-shot performing well'
    )
    print("    Result logged")

    print("\n3. A/B results from database...")
    results = manager.get_ab_results()
    if results:
        for r in results:
            print(
                f"   {r['variant']}: "
                f"overall={r['avg_overall']:.2f}"
            )
    else:
        print("   No results yet (just logged first one)")

    print("\n4. Stats:")
    stats = manager.get_stats()
    print(
        f"   Consistency checks: "
        f"{stats['total_consistency_checks']}"
    )
    print(
        f"   High agreement rate: "
        f"{stats['high_agreement_rate']}%"
    )

    print("\n" + "=" * 60)
    print("=" * 60)