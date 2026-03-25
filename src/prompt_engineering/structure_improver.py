""" Structure Improver - JSON consistency"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List, Tuple, Optional
import json
import re
import logging
from datetime import datetime
from dataclasses import dataclass

from explanation.groq_client import GroqLLMClient
from explanation.response_parser import LLMResponseParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StructureCheckResult:
    """Result of structure check"""
    is_valid: bool
    validity_score: float
    fields_present: List[str]
    fields_missing: List[str]
    fields_invalid: List[str]
    issues: List[str]
    auto_fixed: bool
    fixed_data: Optional[Dict]


class StructureImprover:
    """
    Improves JSON structure consistency

    Features:
    - Enhanced schema validation
    - Auto-fix common issues
    - Structure consistency metrics
    - Improved JSON prompting
    """

    def __init__(self):
        self.groq_client = GroqLLMClient()
        self.parser = LLMResponseParser()

        self.required_fields = {
            'summary': {
                'type': str,
                'min_length': 20,
                'max_length': 200
            },
            'recommendation': {
                'type': str,
                'min_length': 30,
                'max_length': 500
            },
            'risk_level': {
                'type': str,
                'allowed': ['low', 'medium', 'high', 'critical']
            },
            'key_differences': {
                'type': list,
                'min_items': 1,
                'max_items': 5
            },
            'confidence': {
                'type': str,
                'allowed': ['low', 'medium', 'high']
            }
        }

        self.stats = {
            'total_checked': 0,
            'valid': 0,
            'invalid': 0,
            'auto_fixed': 0
        }

    def check_structure(
        self,
        data: Dict,
        attempt_fix: bool = True
    ) -> StructureCheckResult:
        """Check and optionally fix structure"""
        self.stats['total_checked'] += 1
        issues = []
        fields_present = []
        fields_missing = []
        fields_invalid = []

        explanation = data.get('explanation', data)

        for field, rules in self.required_fields.items():
            if field not in explanation:
                fields_missing.append(field)
                issues.append(f'missing_{field}')
                continue

            value = explanation[field]
            fields_present.append(field)

            # Type check
            expected_type = rules['type']
            if not isinstance(value, expected_type):
                if expected_type == str and isinstance(value, (int, float)):
                    explanation[field] = str(value)
                elif expected_type == list and isinstance(value, str):
                    explanation[field] = [value]
                else:
                    fields_invalid.append(field)
                    issues.append(f'wrong_type_{field}')
                    continue

            # Value-specific checks
            if 'allowed' in rules:
                val = str(value).lower()
                if hasattr(value, 'value'):
                    val = value.value
                if val not in rules['allowed']:
                    fields_invalid.append(field)
                    issues.append(f'invalid_value_{field}')

            if 'min_length' in rules and isinstance(value, str):
                if len(value) < rules['min_length']:
                    issues.append(f'{field}_too_short')

            if 'max_length' in rules and isinstance(value, str):
                if len(value) > rules['max_length']:
                    issues.append(f'{field}_too_long')

            if 'min_items' in rules and isinstance(value, list):
                if len(value) < rules['min_items']:
                    issues.append(f'{field}_too_few_items')

        is_valid = (
            len(fields_missing) == 0
            and len(fields_invalid) == 0
            and len(issues) == 0
        )

        validity_score = max(0.0, 1.0 - (
            len(fields_missing) * 0.2
            + len(fields_invalid) * 0.15
            + len([i for i in issues
                   if 'too_short' in i or 'too_long' in i]) * 0.1
        ))

        fixed_data = None
        auto_fixed = False

        if not is_valid and attempt_fix:
            fixed_data, auto_fixed = self._auto_fix(
                explanation, fields_missing, fields_invalid, issues
            )

        if is_valid:
            self.stats['valid'] += 1
        else:
            self.stats['invalid'] += 1
            if auto_fixed:
                self.stats['auto_fixed'] += 1

        return StructureCheckResult(
            is_valid=is_valid,
            validity_score=validity_score,
            fields_present=fields_present,
            fields_missing=fields_missing,
            fields_invalid=fields_invalid,
            issues=issues,
            auto_fixed=auto_fixed,
            fixed_data=fixed_data
        )

    def _auto_fix(
        self,
        data: Dict,
        missing: List[str],
        invalid: List[str],
        issues: List[str]
    ) -> Tuple[Dict, bool]:
        """Auto-fix common structure issues"""
        fixed = data.copy()
        made_fix = False

        # Fix missing fields with defaults
        defaults = {
            'summary': 'Compliance gap identified - review required',
            'recommendation': (
                'Review regulation and update policy accordingly'
            ),
            'risk_level': 'medium',
            'key_differences': ['Gap identified - details unclear'],
            'confidence': 'low'
        }

        for field in missing:
            fixed[field] = defaults[field]
            made_fix = True

        # Fix invalid risk_level
        if 'risk_level' in invalid or 'invalid_value_risk_level' in issues:
            rl = str(fixed.get('risk_level', '')).lower()
            if 'critical' in rl:
                fixed['risk_level'] = 'critical'
            elif 'high' in rl:
                fixed['risk_level'] = 'high'
            elif 'medium' in rl or 'moderate' in rl:
                fixed['risk_level'] = 'medium'
            else:
                fixed['risk_level'] = 'low'
            made_fix = True

        # Fix invalid confidence
        if 'invalid_value_confidence' in issues:
            conf = str(fixed.get('confidence', '')).lower()
            if 'high' in conf:
                fixed['confidence'] = 'high'
            elif 'low' in conf:
                fixed['confidence'] = 'low'
            else:
                fixed['confidence'] = 'medium'
            made_fix = True

        # Fix summary too long
        if 'summary_too_long' in issues:
            fixed['summary'] = fixed['summary'][:197] + "..."
            made_fix = True

        # Fix key_differences not a list
        if not isinstance(fixed.get('key_differences'), list):
            val = fixed.get('key_differences', '')
            fixed['key_differences'] = (
                [str(val)] if val
                else ['Gap identified']
            )
            made_fix = True

        return fixed, made_fix

    def measure_consistency(
        self,
        responses: List[Dict]
    ) -> Dict:
        """Measure structural consistency across responses"""
        if not responses:
            return {}

        total = len(responses)
        valid_count = 0
        field_presence = {
            field: 0 for field in self.required_fields
        }

        for resp in responses:
            result = self.check_structure(resp, attempt_fix=False)
            if result.is_valid:
                valid_count += 1
            for field in result.fields_present:
                if field in field_presence:
                    field_presence[field] += 1

        return {
            'total': total,
            'valid': valid_count,
            'validity_rate': round(
                valid_count / total * 100, 1
            ),
            'field_presence_rates': {
                field: round(count / total * 100, 1)
                for field, count in field_presence.items()
            },
            'target_validity_rate': 95.0,
            'meets_target': (valid_count / total * 100) >= 95.0
        }

    def get_stats(self) -> Dict:
        """Get improver statistics"""
        total = self.stats['total_checked']
        return {
            **self.stats,
            'validity_rate': round(
                self.stats['valid'] / total * 100
                if total > 0 else 0, 1
            ),
            'fix_rate': round(
                self.stats['auto_fixed'] / total * 100
                if total > 0 else 0, 1
            )
        }


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Structure Improver Test")
    print("=" * 60)

    improver = StructureImprover()

    test_cases = [
        {
            'name': 'Perfect structure',
            'data': {
                'summary': 'Policy lacks MFA requirement.',
                'recommendation': (
                    'Implement MFA for all admin accounts.'
                ),
                'risk_level': 'high',
                'key_differences': ['Missing MFA'],
                'confidence': 'high'
            }
        },
        {
            'name': 'Missing fields',
            'data': {
                'summary': 'Gap found.',
                'risk_level': 'medium'
            }
        },
        {
            'name': 'Invalid values',
            'data': {
                'summary': 'Compliance gap identified here.',
                'recommendation': 'Update the policy document.',
                'risk_level': 'VERY HIGH',
                'key_differences': ['Gap 1'],
                'confidence': 'very confident'
            }
        },
        {
            'name': 'Wrong types',
            'data': {
                'summary': 'Gap found in policy.',
                'recommendation': 'Fix the policy now.',
                'risk_level': 'high',
                'key_differences': 'Missing MFA',
                'confidence': 'high'
            }
        }
    ]

    print("\n1. Structure checks with auto-fix...")
    for case in test_cases:
        result = improver.check_structure(case['data'])
        status = "PASS"if result.is_valid else "FAIL"
        fixed = "→ 🔧 Fixed" if result.auto_fixed else ""
        print(
            f"   {status} {case['name']:25} "
            f"Score: {result.validity_score:.2f} {fixed}"
        )
        if result.issues:
            print(f"      Issues: {result.issues[:3]}")

    print("\n2. Consistency measurement...")
    responses = [case['data'] for case in test_cases]
    consistency = improver.measure_consistency(responses)
    print(f"   Validity rate: {consistency['validity_rate']}%")
    print(f"   Target (95%): "
          f"{' Met' if consistency['meets_target'] else ' Not met'}")
    print(f"   Field presence: "
          f"{consistency['field_presence_rates']}")

    print("\n3. Stats:")
    stats = improver.get_stats()
    print(f"   Validity rate: {stats['validity_rate']}%")
    print(f"   Auto-fix rate: {stats['fix_rate']}%")

    print("\n" + "=" * 60)
    print("=" * 60)