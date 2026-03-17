"""Response Parser - Robust JSON extraction from Phi-3 Mini outputs"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, Optional, Tuple
import json
import re
import logging
from dataclasses import dataclass
import time

from explanation.schemas import GapExplanation, SchemaValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """Result of a parsing attempt"""
    success: bool
    data: Optional[Dict]
    method: str
    error: Optional[str]
    raw_text: str
    parse_time_ms: float
    attempts: int


class LLMResponseParser:
    """
    Robust parser for Phi-3 Mini JSON outputs

    5 fallback strategies:
    1. Direct JSON parse
    2. Strip markdown code blocks
    3. Regex JSON extraction
    4. Fix common JSON errors
    5. Field-by-field extraction
    """

    def __init__(self):
        self.validator = SchemaValidator()
        self.parse_stats = {
            'total_attempts': 0,
            'successful_parses': 0,
            'failed_parses': 0,
            'method_success': {
                'direct': 0,
                'markdown_strip': 0,
                'regex_extract': 0,
                'fix_json': 0,
                'fallback': 0
            }
        }

    def parse(
        self,
        raw_text: str,
        attempt_validation: bool = True
    ) -> ParseResult:
        """
        Parse LLM response with multiple fallback strategies

        Args:
            raw_text: Raw text output from Phi-3 Mini
            attempt_validation: Validate against Pydantic schema

        Returns:
            ParseResult with extracted data or error info
        """
        start_time = time.time()
        self.parse_stats['total_attempts'] += 1
        raw_text = raw_text.strip()
        attempts = 0

        # ── Strategy 1: Direct JSON parse ──────────────
        attempts += 1
        try:
            data = json.loads(raw_text)
            elapsed = (time.time() - start_time) * 1000

            if attempt_validation:
                success, validated, error = \
                    self.validator.validate_gap_explanation(data)
                if success:
                    self.parse_stats['successful_parses'] += 1
                    self.parse_stats['method_success']['direct'] += 1
                    return ParseResult(
                        success=True, data=validated.to_dict(),
                        method='direct', error=None,
                        raw_text=raw_text[:200],
                        parse_time_ms=elapsed, attempts=attempts
                    )

            self.parse_stats['successful_parses'] += 1
            self.parse_stats['method_success']['direct'] += 1
            return ParseResult(
                success=True, data=data, method='direct',
                error=None, raw_text=raw_text[:200],
                parse_time_ms=elapsed, attempts=attempts
            )
        except json.JSONDecodeError:
            pass

        # ── Strategy 2: Strip markdown blocks ──────────
        attempts += 1
        try:
            cleaned = self._strip_markdown(raw_text)
            data = json.loads(cleaned)
            elapsed = (time.time() - start_time) * 1000
            self.parse_stats['successful_parses'] += 1
            self.parse_stats['method_success']['markdown_strip'] += 1
            return ParseResult(
                success=True, data=data, method='markdown_strip',
                error=None, raw_text=raw_text[:200],
                parse_time_ms=elapsed, attempts=attempts
            )
        except (json.JSONDecodeError, ValueError):
            pass

        # ── Strategy 3: Regex JSON extraction ──────────
        attempts += 1
        try:
            data = self._extract_json_regex(raw_text)
            if data:
                elapsed = (time.time() - start_time) * 1000
                self.parse_stats['successful_parses'] += 1
                self.parse_stats['method_success']['regex_extract'] += 1
                return ParseResult(
                    success=True, data=data, method='regex_extract',
                    error=None, raw_text=raw_text[:200],
                    parse_time_ms=elapsed, attempts=attempts
                )
        except Exception:
            pass

        # ── Strategy 4: Fix common JSON errors ─────────
        attempts += 1
        try:
            fixed = self._fix_common_json_errors(raw_text)
            data = json.loads(fixed)
            elapsed = (time.time() - start_time) * 1000
            self.parse_stats['successful_parses'] += 1
            self.parse_stats['method_success']['fix_json'] += 1
            return ParseResult(
                success=True, data=data, method='fix_json',
                error=None, raw_text=raw_text[:200],
                parse_time_ms=elapsed, attempts=attempts
            )
        except json.JSONDecodeError:
            pass

        # ── Strategy 5: Field-by-field extraction ──────
        attempts += 1
        data = self._fallback_extraction(raw_text)
        elapsed = (time.time() - start_time) * 1000

        if data and len(data) >= 3:
            self.parse_stats['successful_parses'] += 1
            self.parse_stats['method_success']['fallback'] += 1
            return ParseResult(
                success=True, data=data, method='fallback',
                error="Partial extraction",
                raw_text=raw_text[:200],
                parse_time_ms=elapsed, attempts=attempts
            )

        # ── All strategies failed ───────────────────────
        self.parse_stats['failed_parses'] += 1
        return ParseResult(
            success=False, data=None, method='all_failed',
            error="All parsing strategies failed",
            raw_text=raw_text[:500],
            parse_time_ms=elapsed, attempts=attempts
        )

    def _strip_markdown(self, text: str) -> str:
        """Remove markdown code block markers"""
        patterns = [
            r'^```json\s*', r'^```\s*',
            r'\s*```$', r'^`', r'`$'
        ]
        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE)
        return cleaned.strip()

    def _extract_json_regex(self, text: str) -> Optional[Dict]:
        """Extract JSON object using regex"""
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None

    def _fix_common_json_errors(self, text: str) -> str:
        """Fix common JSON formatting errors from Phi-3"""
        # Remove trailing commas
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        # Fix single quotes to double quotes
        text = re.sub(r"'([^']*)':", r'"\1":', text)
        text = re.sub(r": '([^']*)'", r': "\1"', text)
        # Remove comments
        text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
        return text

    def _fallback_extraction(self, text: str) -> Dict:
        """Last resort: extract fields using regex patterns"""
        result = {}

        patterns = {
            'summary': r'"summary":\s*"([^"]+)"',
            'recommendation': r'"recommendation":\s*"([^"]+)"',
            'risk_level': r'"risk_level":\s*"(\w+)"',
            'confidence': r'"confidence":\s*"(\w+)"'
        }

        for field, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result[field] = match.group(1).lower() \
                    if field in ['risk_level', 'confidence'] \
                    else match.group(1)

        # Extract key_differences
        diff_section = re.search(
            r'"key_differences":\s*\[(.*?)\]',
            text, re.DOTALL | re.IGNORECASE
        )
        if diff_section:
            items = re.findall(r'"([^"]+)"', diff_section.group(1))
            result['key_differences'] = items

        if 'key_differences' not in result:
            result['key_differences'] = ['Extraction incomplete']

        return result

    def parse_with_confidence(
        self,
        raw_text: str,
        original_prompt: str
    ) -> Tuple[Dict, str, float]:
        """
        Parse and assign confidence based on parsing method

        Returns: (data, method, confidence_score)
        """
        result = self.parse(raw_text)

        if not result.success:
            return {
                'summary': 'Parsing failed - manual review required',
                'recommendation': 'Review raw LLM output manually',
                'risk_level': 'medium',
                'key_differences': ['Parsing error'],
                'confidence': 'low'
            }, 'failed', 0.0

        method_confidence = {
            'direct': 1.0,
            'markdown_strip': 0.9,
            'regex_extract': 0.7,
            'fix_json': 0.6,
            'fallback': 0.4
        }

        conf_score = method_confidence.get(result.method, 0.5)
        return result.data, result.method, conf_score

    def get_stats(self) -> Dict:
        """Get parsing statistics"""
        total = self.parse_stats['total_attempts']
        success_rate = (
            self.parse_stats['successful_parses'] / total * 100
            if total > 0 else 0
        )
        return {
            **self.parse_stats,
            'success_rate_percent': round(success_rate, 2)
        }


# Test
if __name__ == "__main__":
    print("=" * 60)
    print(" Response Parser Test")
    print("=" * 60)

    parser = LLMResponseParser()

    test_cases = [
        ("Direct JSON",
         '{"summary": "Test gap found", "recommendation": "Update policy", '
         '"risk_level": "medium", "key_differences": ["A", "B"], '
         '"confidence": "high"}'),

        ("Markdown wrapped",
         '```json\n{"summary": "Test gap found", "recommendation": '
         '"Update policy", "risk_level": "medium", '
         '"key_differences": ["A", "B"], "confidence": "high"}\n```'),

        ("With extra text",
         'Here is my analysis:\n\n{"summary": "Test gap found", '
         '"recommendation": "Update policy", "risk_level": "medium", '
         '"key_differences": ["A", "B"], "confidence": "high"}\n\nDone.'),

        ("Trailing comma",
         '{"summary": "Test", "recommendation": "Do something", '
         '"risk_level": "medium", "key_differences": ["A", "B",], '
         '"confidence": "high",}'),

        ("Partial/malformed",
         'summary: Test gap\nrecommendation: Do something\n'
         'risk_level: medium'),
    ]

    print("\n1. Testing all parsing strategies...")
    for name, text in test_cases:
        result = parser.parse(text)
        status = "PASS" if result.success else "FAIL"
        print(
            f"   {status} {name:20} → {result.method:15} "
            f"({result.parse_time_ms:.1f}ms)"
        )

    print("\n2. Testing complete failure...")
    fail = parser.parse("This is definitely not JSON!!!")
    print(f"   Success: {fail.success} (expected False)")

    print("\n3. Parser statistics:")
    stats = parser.get_stats()
    print(f"   Total: {stats['total_attempts']}")
    print(f"   Success rate: {stats['success_rate_percent']}%")
    print(f"   By method: {stats['method_success']}")

    print("\n" + "=" * 60)
    print("=" * 60)