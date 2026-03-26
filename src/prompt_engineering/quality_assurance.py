"""Quality Assurance System"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List, Optional
import json
import sqlite3
import logging
from datetime import datetime
from dataclasses import dataclass

from prompt_engineering.quality_analyzer import PromptQualityAnalyzer
from prompt_engineering.structure_improver import StructureImprover
from prompt_engineering.refinement_pipeline import RefinementPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GoldenSetEntry:
    """A manually reviewed gap-explanation pair"""
    entry_id: str
    regulation_text: str
    policy_text: str
    expected_risk_level: str
    expected_has_specific_action: bool
    notes: str
    added_at: str


class QualityAssuranceSystem:
    """
    Complete QA system for explanation quality

    Features:
    - Golden set management
    - Automated quality checks
    - Quality dashboard metrics
    - Low confidence flagging
    - Quality trend tracking
    """

    def __init__(
        self,
        db_path: str = "data/qa_system.db"
    ):
        self.db_path = db_path
        self.analyzer = PromptQualityAnalyzer()
        self.structure_improver = StructureImprover()
        self.pipeline = RefinementPipeline()
        self._init_db()
        self._populate_golden_set()

    def _init_db(self):
        """Initialize QA database"""
        import os
        os.makedirs(
            str(Path(self.db_path).parent),
            exist_ok=True
        )
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS golden_set (
                    entry_id TEXT PRIMARY KEY,
                    regulation_text TEXT,
                    policy_text TEXT,
                    expected_risk_level TEXT,
                    expected_has_specific_action INTEGER,
                    notes TEXT,
                    added_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS qa_results (
                    id TEXT PRIMARY KEY,
                    run_id TEXT,
                    entry_id TEXT,
                    overall_score REAL,
                    accuracy_score REAL,
                    actionability_score REAL,
                    structure_score REAL,
                    risk_correct INTEGER,
                    passed INTEGER,
                    issues TEXT,
                    tested_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_trends (
                    id TEXT PRIMARY KEY,
                    run_id TEXT,
                    prompt_version TEXT,
                    avg_overall REAL,
                    pass_rate REAL,
                    golden_accuracy REAL,
                    measured_at TEXT
                )
            """)
            conn.commit()

    def _populate_golden_set(self):
        """Add golden set examples if empty"""
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM golden_set"
            ).fetchone()[0]

            if count > 0:
                return

            golden_examples = [
                (
                    'gs_001',
                    'All administrative accounts must use '
                    'multi-factor authentication.',
                    'Users must create strong passwords '
                    'with minimum 12 characters.',
                    'high',
                    1,
                    'Clear MFA vs password-only gap'
                ),
                (
                    'gs_002',
                    'Personal data must be encrypted '
                    'using AES-256 at rest and in transit.',
                    'We use industry standard encryption '
                    'for our databases.',
                    'high',
                    1,
                    'Partial coverage - missing in-transit'
                ),
                (
                    'gs_003',
                    'Data breach notification must occur '
                    'within 72 hours of discovery.',
                    'We have an incident response team '
                    'that handles security events.',
                    'critical',
                    1,
                    'Missing specific 72-hour requirement'
                ),
                (
                    'gs_004',
                    'Employee security awareness training '
                    'must be completed annually.',
                    'We provide onboarding training for '
                    'new employees on security.',
                    'medium',
                    1,
                    'Missing annual recurrence requirement'
                ),
                (
                    'gs_005',
                    'Access to sensitive systems must be '
                    'reviewed quarterly.',
                    'Access rights are reviewed during '
                    'annual performance reviews.',
                    'medium',
                    1,
                    'Frequency mismatch: annual vs quarterly'
                ),
            ]

            for ex in golden_examples:
                conn.execute("""
                    INSERT OR IGNORE INTO golden_set VALUES
                    (?, ?, ?, ?, ?, ?, ?)
                """, (*ex, datetime.now().isoformat()))

            conn.commit()
            logger.info(
                f"Populated golden set with "
                f"{len(golden_examples)} examples"
            )

    def get_golden_set(self) -> List[GoldenSetEntry]:
        """Get all golden set entries"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM golden_set"
            ).fetchall()

        return [
            GoldenSetEntry(
                entry_id=row[0],
                regulation_text=row[1],
                policy_text=row[2],
                expected_risk_level=row[3],
                expected_has_specific_action=bool(row[4]),
                notes=row[5],
                added_at=row[6]
            )
            for row in rows
        ]

    def evaluate_on_golden_set(
        self,
        explanations: List[Dict],
        run_id: str,
        prompt_version: str = "current"
    ) -> Dict:
        """
        Evaluate explanations against golden set

        Args:
            explanations: Generated explanations to evaluate
            run_id: Identifier for this evaluation run
            prompt_version: Which prompt version was used

        Returns:
            Evaluation report
        """
        golden_set = self.get_golden_set()

        if len(explanations) < len(golden_set):
            logger.warning(
                f"Only {len(explanations)} explanations "
                f"for {len(golden_set)} golden examples"
            )

        results = []
        risk_correct = 0
        action_correct = 0

        for i, (exp, golden) in enumerate(
            zip(explanations, golden_set)
        ):
            exp_data = exp.get('explanation', exp)

            # Check risk level correctness
            actual_risk = str(
                exp_data.get('risk_level', '')
            ).lower()
            if hasattr(actual_risk, 'value'):
                actual_risk = actual_risk.value
            expected_risk = golden.expected_risk_level.lower()

            risk_match = actual_risk == expected_risk
            if risk_match:
                risk_correct += 1

            # Check actionability
            rec = str(
                exp_data.get('recommendation', '')
            ).lower()
            has_action = any(
                w in rec
                for w in self.pipeline.quality_analyzer
                .action_words
            )
            if has_action == golden.expected_has_specific_action:
                action_correct += 1

            # Quality score
            quality = self.analyzer.analyze(
                exp,
                golden.regulation_text,
                golden.policy_text
            )

            result = {
                'entry_id': golden.entry_id,
                'risk_correct': risk_match,
                'expected_risk': expected_risk,
                'actual_risk': actual_risk,
                'action_correct': has_action,
                'quality_score': quality.overall_score,
                'passed': quality.passed,
                'issues': quality.issues
            }
            results.append(result)

            # Save to database
            self._save_qa_result(run_id, result, quality)

        total = len(results)
        passed = sum(1 for r in results if r['passed'])

        report = {
            'run_id': run_id,
            'prompt_version': prompt_version,
            'total_evaluated': total,
            'passed': passed,
            'pass_rate': round(
                passed / total * 100 if total > 0 else 0, 1
            ),
            'risk_accuracy': round(
                risk_correct / total * 100
                if total > 0 else 0, 1
            ),
            'action_accuracy': round(
                action_correct / total * 100
                if total > 0 else 0, 1
            ),
            'avg_quality_score': round(
                sum(r['quality_score'] for r in results)
                / total if total > 0 else 0, 3
            ),
            'results': results,
            'evaluated_at': datetime.now().isoformat()
        }

        # Save trend
        self._save_trend(run_id, prompt_version, report)

        return report

    def run_automated_checks(
        self,
        explanations: List[Dict]
    ) -> Dict:
        """
        Run automated QA checks on batch of explanations

        Checks:
        - Length compliance
        - Keyword presence
        - Risk distribution
        - Low confidence flagging
        """
        total = len(explanations)
        if total == 0:
            return {}

        length_ok = 0
        keyword_ok = 0
        low_confidence_flags = []
        risk_dist = {
            'low': 0, 'medium': 0,
            'high': 0, 'critical': 0
        }

        required_keywords = [
            'policy', 'regulation', 'compliance',
            'requirement', 'implement', 'update',
            'establish', 'create', 'define'
        ]

        for i, exp in enumerate(explanations):
            exp_data = exp.get('explanation', exp)

            # Length check
            summary_len = len(
                str(exp_data.get('summary', ''))
            )
            rec_len = len(
                str(exp_data.get('recommendation', ''))
            )
            if (20 <= summary_len <= 200
                    and 30 <= rec_len <= 500):
                length_ok += 1

            # Keyword check
            combined = (
                str(exp_data.get('summary', '')).lower()
                + ' '
                + str(exp_data.get('recommendation', '')).lower()
            )
            has_keyword = any(
                kw in combined for kw in required_keywords
            )
            if has_keyword:
                keyword_ok += 1

            # Risk distribution
            risk = str(
                exp_data.get('risk_level', 'medium')
            ).lower()
            if hasattr(risk, 'value'):
                risk = risk.value
            if risk in risk_dist:
                risk_dist[risk] += 1

            # Low confidence flagging
            conf = str(
                exp_data.get('confidence', 'medium')
            ).lower()
            if hasattr(conf, 'value'):
                conf = conf.value
            if conf == 'low':
                low_confidence_flags.append(i)

        # Check risk distribution is reasonable
        all_same_risk = any(
            v == total for v in risk_dist.values()
        )

        return {
            'total': total,
            'length_compliance_rate': round(
                length_ok / total * 100, 1
            ),
            'keyword_presence_rate': round(
                keyword_ok / total * 100, 1
            ),
            'risk_distribution': risk_dist,
            'risk_distribution_healthy': not all_same_risk,
            'low_confidence_count': len(low_confidence_flags),
            'low_confidence_indices': low_confidence_flags,
            'low_confidence_rate': round(
                len(low_confidence_flags) / total * 100, 1
            ),
            'checks_passed': (
                length_ok / total >= 0.8
                and keyword_ok / total >= 0.7
                and not all_same_risk
            )
        }

    def get_quality_dashboard(self) -> Dict:
        """Get quality metrics dashboard"""
        with sqlite3.connect(self.db_path) as conn:
            # Overall metrics
            overall = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    AVG(overall_score) as avg_score,
                    SUM(passed) as total_passed,
                    AVG(risk_correct) as risk_accuracy
                FROM qa_results
            """).fetchone()

            # Recent trend
            trends = conn.execute("""
                SELECT prompt_version,
                       avg_overall,
                       pass_rate,
                       golden_accuracy,
                       measured_at
                FROM quality_trends
                ORDER BY measured_at DESC
                LIMIT 5
            """).fetchall()

        return {
            'overall_metrics': {
                'total_evaluated': overall[0] or 0,
                'avg_quality_score': round(
                    overall[1] or 0, 3
                ),
                'pass_rate': round(
                    (overall[2] or 0) / max(overall[0] or 1, 1)
                    * 100, 1
                ),
                'risk_accuracy': round(
                    (overall[3] or 0) * 100, 1
                )
            },
            'recent_trends': [
                {
                    'version': t[0],
                    'avg_overall': t[1],
                    'pass_rate': t[2],
                    'golden_accuracy': t[3],
                    'measured_at': t[4]
                }
                for t in trends
            ],
            'golden_set_size': len(self.get_golden_set()),
            'dashboard_generated_at': datetime.now().isoformat()
        }

    def flag_low_confidence(
        self,
        explanations: List[Dict],
        confidence_threshold: float = 0.6
    ) -> List[int]:
        """Flag explanations needing human review"""
        flagged = []

        for i, exp in enumerate(explanations):
            quality = self.analyzer.analyze(exp)
            if quality.overall_score < confidence_threshold:
                flagged.append({
                    'index': i,
                    'score': quality.overall_score,
                    'issues': quality.issues
                })

        return flagged

    def _save_qa_result(
        self,
        run_id: str,
        result: Dict,
        quality
    ):
        """Save QA result to database"""
        import hashlib
        result_id = hashlib.md5(
            f"{run_id}{result['entry_id']}"
            f"{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO qa_results VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result_id, run_id,
                result['entry_id'],
                quality.overall_score,
                quality.accuracy_score,
                quality.actionability_score,
                quality.structure_score,
                int(result['risk_correct']),
                int(result['passed']),
                json.dumps(result['issues']),
                datetime.now().isoformat()
            ))
            conn.commit()

    def _save_trend(
        self,
        run_id: str,
        prompt_version: str,
        report: Dict
    ):
        """Save quality trend"""
        import hashlib
        trend_id = hashlib.md5(
            f"{run_id}{datetime.now().isoformat()}"
            .encode()
        ).hexdigest()[:16]

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO quality_trends VALUES
                (?, ?, ?, ?, ?, ?, ?)
            """, (
                trend_id, run_id, prompt_version,
                report['avg_quality_score'],
                report['pass_rate'],
                report['risk_accuracy'],
                datetime.now().isoformat()
            ))
            conn.commit()

import os
if os.path.exists("data/qa_system.db"):
    os.remove("data/qa_system.db")            

# Test
if __name__ == "__main__":
    print("=" * 60)
    print(" Quality Assurance System Test")
    print("=" * 60)

    qa = QualityAssuranceSystem()

    print("\n1. Golden set loaded...")
    golden = qa.get_golden_set()
    print(f"   Golden set size: {len(golden)} examples")
    for g in golden[:3]:
        print(
            f"   - {g.entry_id}: "
            f"Expected risk={g.expected_risk_level}"
        )

    print("\n2. Automated checks on mock explanations...")
    mock_explanations = [
        {
            'explanation': {
                'summary': (
                    'Policy lacks MFA requirement '
                    'for admin access.'
                ),
                'recommendation': (
                    'Implement MFA for all admin accounts.'
                ),
                'risk_level': 'high',
                'key_differences': ['Missing MFA'],
                'confidence': 'high'
            }
        },
        {
            'explanation': {
                'summary': 'Gap found.',
                'recommendation': 'Fix it.',
                'risk_level': 'critical',
                'key_differences': [],
                'confidence': 'low'
            }
        },
        {
            'explanation': {
                'summary': (
                    'Encryption policy missing '
                    'for data at rest.'
                ),
                'recommendation': (
                    'Define encryption standards '
                    'and implement AES-256.'
                ),
                'risk_level': 'high',
                'key_differences': [
                    'No encryption standard',
                    'Missing key management'
                ],
                'confidence': 'medium'
            }
        }
    ]

    checks = qa.run_automated_checks(mock_explanations)
    print(
        f"   Length compliance: "
        f"{checks['length_compliance_rate']}%"
    )
    print(
        f"   Keyword presence: "
        f"{checks['keyword_presence_rate']}%"
    )
    print(
        f"   Risk distribution: "
        f"{checks['risk_distribution']}"
    )
    print(
        f"   Low confidence: "
        f"{checks['low_confidence_count']} flagged"
    )
    print(
        f"   All checks passed: "
        f"{'PASS' if checks['checks_passed'] else 'FAIL'}"
    )

    print("\n3. Golden set evaluation...")
    eval_report = qa.evaluate_on_golden_set(
        mock_explanations[:3],
        run_id='test_run_001',
        prompt_version='v3_fewshot'
    )
    print(f"   Pass rate: {eval_report['pass_rate']}%")
    print(f"   Risk accuracy: {eval_report['risk_accuracy']}%")
    print(
        f"   Avg quality: "
        f"{eval_report['avg_quality_score']:.2f}"
    )

    print("\n4. Quality dashboard...")
    dashboard = qa.get_quality_dashboard()
    metrics = dashboard['overall_metrics']
    print(
        f"   Total evaluated: "
        f"{metrics['total_evaluated']}"
    )
    print(
        f"   Avg quality score: "
        f"{metrics['avg_quality_score']:.2f}"
    )
    print(f"   Pass rate: {metrics['pass_rate']}%")

    print("\n5. Low confidence flagging...")
    flagged = qa.flag_low_confidence(
        mock_explanations, confidence_threshold=0.6
    )
    print(f"   Flagged for review: {len(flagged)} items")
    for f in flagged:
        print(
            f"   - Index {f['index']}: "
            f"score={f['score']:.2f}"
        )

    print("\n" + "=" * 60)
