"""Response Parser - JSON extraction from Groq/Llama3 outputs"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, Optional, Tuple
import json
import re
import logging
from dataclasses import dataclass
import time

from explanation.schemas import SchemaValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """Result of parsing attempt"""
    success: bool
    data: Optional[Dict]
    method: str
    error: Optional[str]
    raw_text: str
    parse_time_ms: float
    attempts: int


class LLMResponseParser:
    """
    Robust parser for Llama3/Groq JSON outputs

    Note: Llama3 via Groq produces cleaner JSON than Phi-3,
    so strategies 1-2 handle ~95% of cases.
    Strategies 3-5 are safety nets.
    """

    def __init__(self):
        self.validator = SchemaValidator()
        self.stats = {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'by_method': {
                'direct': 0, 'markdown_strip': 0,
                'regex_extract': 0, 'fix_json': 0, 'fallback': 0
            }
        }

    def parse(
        self,
        raw_text: str,
        attempt_validation: bool = True
    ) -> ParseResult:
        """
        Parse LLM response with fallback strategies

        Args:
            raw_text: Raw text from Groq API
            attempt_validation: Validate against schema

        Returns:
            ParseResult
        """
        start = time.time()
        self.stats['total'] += 1
        text = raw_text.strip()
        attempts = 0

        # Strategy 1: Direct JSON (Llama3 usually nails this)
        attempts += 1
        try:
            data = json.loads(text)
            elapsed = (time.time() - start) * 1000
            if attempt_validation:
                ok, validated, err = \
                    self.validator.validate_gap_explanation(data)
                if ok:
                    self.stats['successful'] += 1
                    self.stats['by_method']['direct'] += 1
                    return ParseResult(
                        True, validated.to_dict(), 'direct',
                        None, text[:200], elapsed, attempts
                    )
            self.stats['successful'] += 1
            self.stats['by_method']['direct'] += 1
            return ParseResult(
                True, data, 'direct',
                None, text[:200], elapsed, attempts
            )
        except json.JSONDecodeError:
            pass

        # Strategy 2: Strip markdown (common with Llama3)
        attempts += 1
        try:
            cleaned = re.sub(
                r'^```(?:json)?\s*|\s*```$', '', text,
                flags=re.MULTILINE
            ).strip()
            data = json.loads(cleaned)
            elapsed = (time.time() - start) * 1000
            self.stats['successful'] += 1
            self.stats['by_method']['markdown_strip'] += 1
            return ParseResult(
                True, data, 'markdown_strip',
                None, text[:200], elapsed, attempts
            )
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 3: Extract JSON block with regex
        attempts += 1
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                elapsed = (time.time() - start) * 1000
                self.stats['successful'] += 1
                self.stats['by_method']['regex_extract'] += 1
                return ParseResult(
                    True, data, 'regex_extract',
                    None, text[:200], elapsed, attempts
                )
        except Exception:
            pass

        # Strategy 4: Fix common JSON errors
        attempts += 1
        try:
            fixed = text
            fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
            fixed = re.sub(r"'([^']*)':", r'"\1":', fixed)
            fixed = re.sub(r': \'([^\']*)\'', r': "\1"', fixed)
            data = json.loads(fixed)
            elapsed = (time.time() - start) * 1000
            self.stats['successful'] += 1
            self.stats['by_method']['fix_json'] += 1
            return ParseResult(
                True, data, 'fix_json',
                None, text[:200], elapsed, attempts
            )
        except Exception:
            pass

        # Strategy 5: Field extraction (last resort)
        attempts += 1
        data = self._extract_fields(text)
        elapsed = (time.time() - start) * 1000

        if data and len(data) >= 3:
            self.stats['successful'] += 1
            self.stats['by_method']['fallback'] += 1
            return ParseResult(
                True, data, 'fallback',
                "Partial extraction", text[:200], elapsed, attempts
            )

        self.stats['failed'] += 1
        return ParseResult(
            False, None, 'all_failed',
            "All strategies failed", text[:500], elapsed, attempts
        )

    def _extract_fields(self, text: str) -> Dict:
        """Extract individual fields using regex"""
        result = {}

        fields = {
            'summary': r'"summary":\s*"([^"]+)"',
            'recommendation': r'"recommendation":\s*"([^"]+)"',
            'risk_level': r'"risk_level":\s*"(\w+)"',
            'confidence': r'"confidence":\s*"(\w+)"'
        }

        for field, pattern in fields.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                val = match.group(1)
                result[field] = (
                    val.lower() if field in ['risk_level', 'confidence']
                    else val
                )

        diff_match = re.search(
            r'"key_differences":\s*\[(.*?)\]', text, re.DOTALL
        )
        if diff_match:
            result['key_differences'] = re.findall(
                r'"([^"]+)"', diff_match.group(1)
            )
        else:
            result['key_differences'] = ['Manual review needed']

        return result

    def parse_with_confidence(
        self,
        raw_text: str
    ) -> Tuple[Dict, str, float]:
        """
        Parse and return confidence score

        Returns: (data, method, confidence_0_to_1)
        """
        result = self.parse(raw_text)

        if not result.success:
            return {
                'summary': 'Parsing failed',
                'recommendation': 'Manual review required',
                'risk_level': 'medium',
                'key_differences': ['Parse error'],
                'confidence': 'low'
            }, 'failed', 0.0

        scores = {
            'direct': 1.0, 'markdown_strip': 0.9,
            'regex_extract': 0.7, 'fix_json': 0.6, 'fallback': 0.4
        }

        return result.data, result.method, scores.get(result.method, 0.5)

    def get_stats(self) -> Dict:
        total = self.stats['total']
        return {
            **self.stats,
            'success_rate': round(
                self.stats['successful'] / total * 100
                if total > 0 else 0, 1
            )
        }


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Day 32: Response Parser Test")
    print("=" * 60)

    parser = LLMResponseParser()

    test_cases = [
        ("Direct JSON",
         '{"summary": "MFA gap found", "recommendation": "Add MFA policy",'
         '"risk_level": "high", "key_differences": ["Missing MFA"],'
         '"confidence": "high"}'),

        ("Markdown wrapped",
         '```json\n{"summary": "MFA gap found", '
         '"recommendation": "Add MFA policy", '
         '"risk_level": "high", "key_differences": ["Missing MFA"],'
         '"confidence": "high"}\n```'),

        ("Extra text around JSON",
         'Here is my analysis:\n'
         '{"summary": "MFA gap found", "recommendation": "Add MFA",'
         '"risk_level": "high", "key_differences": ["Missing MFA"],'
         '"confidence": "high"}\nHope this helps!'),

        ("Trailing comma",
         '{"summary": "Gap found", "recommendation": "Fix it",'
         '"risk_level": "medium", "key_differences": ["A", "B",],'
         '"confidence": "high",}'),

        ("Not JSON at all",
         'The policy is missing several key requirements.'),
    ]

    print("\n1. Testing parsing strategies...")
    for name, text in test_cases:
        result = parser.parse(text)
        status = "PASS" if result.success else "FAIL"
        print(
            f"   {status} {name:25} → {result.method:15} "
            f"({result.parse_time_ms:.1f}ms)"
        )

    print("\n2. Stats:")
    stats = parser.get_stats()
    print(f"   Success rate: {stats['success_rate']}%")
    print(f"   By method: {stats['by_method']}")

    print("\n" + "=" * 60)
    print("=" * 60)