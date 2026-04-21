"""
Coverage Checker - The core compliance verification engine.

This replaces pure cosine similarity with actual compliance checking.
Instead of asking "are these texts similar?", it asks
"does this policy explicitly fulfill this specific obligation?"

This is the biggest architectural improvement in the whole project.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import re
import logging
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoverageType(Enum):
    EXPLICIT = "explicit"      # Policy directly addresses with same specificity
    IMPLICIT = "implicit"      # Policy covers spirit but not letter
    PARTIAL = "partial"        # Covers some but not all specific requirements
    NONE = "none"              # No relevant coverage found
    NOT_APPLICABLE = "not_applicable"  # Chunk is definition/example, skip


@dataclass
class CoverageResult:
    """Result of checking one obligation against one policy"""
    regulation_chunk_id: str
    policy_chunk_id: str
    regulation_source: str
    policy_source: str

    # Core result
    coverage_type: CoverageType
    is_covered: bool
    specific_requirements_met: bool

    # Details
    what_policy_covers: str
    what_is_missing: str
    confidence: str  # high/medium/low

    # Scores (kept for backward compatibility)
    similarity_score: float  # original bi/cross encoder score
    compliance_score: float  # new: 0.0-1.0 based on actual coverage

    # Explanation
    reasoning: str
    regulation_text: str
    policy_text: str

    def to_dict(self) -> Dict:
        return {
            "regulation_chunk_id": self.regulation_chunk_id,
            "policy_chunk_id": self.policy_chunk_id,
            "regulation_source": self.regulation_source,
            "policy_source": self.policy_source,
            "coverage_type": self.coverage_type.value,
            "is_covered": self.is_covered,
            "specific_requirements_met": self.specific_requirements_met,
            "what_policy_covers": self.what_policy_covers,
            "what_is_missing": self.what_is_missing,
            "confidence": self.confidence,
            "similarity_score": round(self.similarity_score, 3),
            "compliance_score": round(self.compliance_score, 3),
            "reasoning": self.reasoning,
            "regulation_text_preview": self.regulation_text[:200],
            "policy_text_preview": self.policy_text[:200]
        }


COVERAGE_CHECK_PROMPT = """You are a senior compliance auditor at an Indian fintech regulator.

Your task: Determine whether a company policy EXPLICITLY satisfies a regulatory obligation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGULATORY OBLIGATION:
{regulation_text}

Source: {regulation_source}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPANY POLICY:
{policy_text}

Source: {policy_source}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Answer these questions carefully:
1. Does the policy EXPLICITLY address this requirement?
2. Are ALL specific requirements (timeframes, standards, thresholds) covered?
3. What exactly does the policy cover?
4. What specific elements are missing?

Definitions:
- "explicit": Policy directly states the same requirement with similar specificity
- "implicit": Policy addresses the concept but not the specific requirement
- "partial": Policy covers some elements but misses specific requirements
- "none": Policy has no relevant coverage of this requirement

Respond ONLY with this JSON:
{{
    "coverage_type": "explicit|implicit|partial|none",
    "is_covered": true/false,
    "specific_requirements_met": true/false,
    "what_policy_covers": "exact quote or description from policy text (max 150 chars)",
    "what_is_missing": "specific missing element (null if fully covered)",
    "confidence": "high|medium|low",
    "reasoning": "1-2 sentence explanation of your decision"
}}"""


# Compliance score mapping
COVERAGE_SCORES = {
    CoverageType.EXPLICIT: 1.0,
    CoverageType.IMPLICIT: 0.65,
    CoverageType.PARTIAL: 0.40,
    CoverageType.NONE: 0.05,
    CoverageType.NOT_APPLICABLE: None  # Skip
}

# Whether specific requirements being met changes the score
SPECIFICS_BONUS = 0.10


class CoverageChecker:
    """
    Checks whether company policies actually cover regulatory obligations.
    
    Uses Groq LLM with a compliance-specific prompt. Falls back to
    heuristic scoring if LLM is unavailable.
    """

    def __init__(self, groq_client=None, use_cache: bool = True):
        self.groq_client = groq_client
        self.use_cache = use_cache
        self._cache: Dict[str, Dict] = {}

        self.stats = {
            'total_checked': 0,
            'explicit': 0,
            'implicit': 0,
            'partial': 0,
            'none': 0,
            'llm_calls': 0,
            'cache_hits': 0,
            'fallback_used': 0
        }

    def _make_cache_key(self, reg_text: str, pol_text: str) -> str:
        import hashlib
        combined = f"{reg_text[:200]}|||{pol_text[:200]}"
        return hashlib.md5(combined.encode()).hexdigest()

    def check_coverage(
        self,
        regulation_text: str,
        policy_text: str,
        regulation_chunk_id: str = "",
        policy_chunk_id: str = "",
        regulation_source: str = "",
        policy_source: str = "",
        similarity_score: float = 0.0,
        delay_seconds: float = 2.0
    ) -> CoverageResult:
        """
        Check if policy covers regulation. This is your main method.
        
        Args:
            regulation_text: The regulation requirement text
            policy_text: The company policy text
            regulation_chunk_id: ID for tracking
            policy_chunk_id: ID for tracking
            regulation_source: Source document name
            policy_source: Policy document name
            similarity_score: Original bi/cross encoder score (kept for reference)
            delay_seconds: API rate limiting delay
        
        Returns:
            CoverageResult with compliance_score 0.0-1.0
        """
        self.stats['total_checked'] += 1

        # Check cache
        cache_key = self._make_cache_key(regulation_text, policy_text)
        if self.use_cache and cache_key in self._cache:
            self.stats['cache_hits'] += 1
            cached = self._cache[cache_key]
            # Rebuild CoverageResult from cached data
            return self._result_from_cache(
                cached, regulation_chunk_id, policy_chunk_id,
                regulation_source, policy_source,
                similarity_score, regulation_text, policy_text
            )

        # Use LLM if available
        if self.groq_client is not None:
            result_data = self._check_with_llm(
                regulation_text, policy_text,
                regulation_source, policy_source,
                delay_seconds
            )
        else:
            result_data = self._check_with_heuristics(
                regulation_text, policy_text
            )

        # Parse coverage type
        coverage_type_str = result_data.get("coverage_type", "none").lower()
        try:
            coverage_type = CoverageType(coverage_type_str)
        except ValueError:
            coverage_type = CoverageType.NONE

        # Calculate compliance score
        base_score = COVERAGE_SCORES.get(coverage_type, 0.05)
        if result_data.get("specific_requirements_met", False):
            base_score = min(base_score + SPECIFICS_BONUS, 1.0)

        # Update stats
        self.stats[coverage_type.value] = (
            self.stats.get(coverage_type.value, 0) + 1
        )

        result = CoverageResult(
            regulation_chunk_id=regulation_chunk_id,
            policy_chunk_id=policy_chunk_id,
            regulation_source=regulation_source,
            policy_source=policy_source,
            coverage_type=coverage_type,
            is_covered=result_data.get("is_covered", False),
            specific_requirements_met=result_data.get(
                "specific_requirements_met", False
            ),
            what_policy_covers=result_data.get("what_policy_covers", ""),
            what_is_missing=result_data.get("what_is_missing", ""),
            confidence=result_data.get("confidence", "low"),
            similarity_score=similarity_score,
            compliance_score=base_score,
            reasoning=result_data.get("reasoning", ""),
            regulation_text=regulation_text,
            policy_text=policy_text
        )

        # Cache result
        if self.use_cache:
            self._cache[cache_key] = result_data

        return result

    def _check_with_llm(
        self,
        regulation_text: str,
        policy_text: str,
        regulation_source: str,
        policy_source: str,
        delay_seconds: float
    ) -> Dict:
        """Call Groq LLM for coverage check"""
        time.sleep(delay_seconds)  # Rate limiting

        try:
            prompt = COVERAGE_CHECK_PROMPT.format(
                regulation_text=regulation_text[:800],
                policy_text=policy_text[:800],
                regulation_source=regulation_source,
                policy_source=policy_source
            )

            response = self.groq_client.generate(
                prompt=prompt,
                system_prompt=(
                    "You are a compliance auditor. "
                    "Respond with valid JSON only. "
                    "Be specific and cite exact text when possible."
                ),
                temperature=0.0,
                max_tokens=400
            )

            self.stats['llm_calls'] += 1

            text = response.text.strip()
            text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=re.MULTILINE)
            return json.loads(text)

        except Exception as e:
            logger.warning(f"LLM coverage check failed: {e}")
            self.stats['fallback_used'] += 1
            return self._check_with_heuristics(regulation_text, policy_text)

    def _check_with_heuristics(
        self,
        regulation_text: str,
        policy_text: str
    ) -> Dict:
        """
        Heuristic fallback - better than pure cosine similarity
        but worse than LLM.
        
        Checks for:
        1. Key obligation verbs from regulation in policy
        2. Subject matter overlap
        3. Specific requirements (timeframes, standards)
        """
        self.stats['fallback_used'] += 1

        reg_lower = regulation_text.lower()
        pol_lower = policy_text.lower()

        # Extract key terms from regulation
        obligation_verbs = re.findall(
            r'\b(shall|must|required|mandatory|prohibited)\b',
            reg_lower
        )

        # Check for timeframe specifics in regulation
        timeframe_units = re.findall(
            r'(\d+\s*(?:hour|day|week|month|year|minute)s?)',
            reg_lower
            )
        # Check if policy mentions same time units (not exact numbers)
        time_unit_words = re.findall(
            r'(hour|day|week|month|year|minute)',
            reg_lower
            )
        timeframes_in_policy = (
            any(unit in pol_lower for unit in time_unit_words)
            if time_unit_words else True
            )

        # Check for technical standards
        standards = re.findall(
            r'\b(aes-\d+|tls|ssl|sha-\d+|rsa|mfa|2fa|otp)\b',
            reg_lower
        )
        standards_in_policy = all(s in pol_lower for s in standards) if standards else True

        # Word overlap
        reg_words = set(re.findall(r'\b\w{5,}\b', reg_lower))
        pol_words = set(re.findall(r'\b\w{5,}\b', pol_lower))
        overlap = len(reg_words & pol_words) / max(len(reg_words), 1)

        # Determine coverage
        has_obligation = len(obligation_verbs) > 0
        has_overlap = overlap > 0.15
        has_specifics = timeframes_in_policy and standards_in_policy

        if has_overlap and has_specifics and overlap > 0.30:
            coverage = "explicit"
            is_covered = True
            specifics_met = True
        elif has_overlap and overlap > 0.20:
            coverage = "partial"
            is_covered = True
            specifics_met = False
        elif has_overlap:
            coverage = "implicit"
            is_covered = False
            specifics_met = False
        else:
            coverage = "none"
            is_covered = False
            specifics_met = False

        missing = []
        if time_unit_words and not timeframes_in_policy:
            missing.append(f"Timeframe requirement: {time_unit_words[0]}")
            if standards and not standards_in_policy:
                missing.append(f"Technical standard: {', '.join(standards)}")
                if not has_overlap:
                    missing.append("No subject matter overlap found")

        return {
            "coverage_type": coverage,
            "is_covered": is_covered,
            "specific_requirements_met": specifics_met,
            "what_policy_covers": f"~{overlap*100:.0f}% keyword overlap" if has_overlap else "Nothing relevant",
            "what_is_missing": "; ".join(missing) if missing else None,
            "confidence": "medium" if has_overlap else "high",
            "reasoning": f"Heuristic: {overlap*100:.0f}% overlap, "
                        f"timeframes {'✓' if timeframes_in_policy else '✗'}, "
                        f"standards {'✓' if standards_in_policy else '✗'}"
        }

    def _result_from_cache(
        self, cached: Dict, reg_id: str, pol_id: str,
        reg_src: str, pol_src: str, sim_score: float,
        reg_text: str, pol_text: str
    ) -> CoverageResult:
        """Reconstruct CoverageResult from cache"""
        coverage_type_str = cached.get("coverage_type", "none")
        try:
            coverage_type = CoverageType(coverage_type_str)
        except ValueError:
            coverage_type = CoverageType.NONE

        base_score = COVERAGE_SCORES.get(coverage_type, 0.05)
        if cached.get("specific_requirements_met", False):
            base_score = min(base_score + SPECIFICS_BONUS, 1.0)

        return CoverageResult(
            regulation_chunk_id=reg_id,
            policy_chunk_id=pol_id,
            regulation_source=reg_src,
            policy_source=pol_src,
            coverage_type=coverage_type,
            is_covered=cached.get("is_covered", False),
            specific_requirements_met=cached.get(
                "specific_requirements_met", False),
            what_policy_covers=cached.get("what_policy_covers", ""),
            what_is_missing=cached.get("what_is_missing", ""),
            confidence=cached.get("confidence", "medium"),
            similarity_score=sim_score,
            compliance_score=base_score,
            reasoning=cached.get("reasoning", ""),
            regulation_text=reg_text,
            policy_text=pol_text
        )

    def get_stats(self) -> Dict:
        total = self.stats['total_checked']
        return {
            **self.stats,
            'explicit_rate': round(
                self.stats['explicit'] / total * 100 if total > 0 else 0, 1
            ),
            'none_rate': round(
                self.stats['none'] / total * 100 if total > 0 else 0, 1
            )
        }


# ── TEST ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    checker = CoverageChecker(groq_client=None)  # heuristic mode

    tests = [
        # Should be EXPLICIT - direct match
        (
            "In the event of a personal data breach, the Data Fiduciary "
            "shall notify the Board within 72 hours of discovery.",
            "TMF shall report any personal data breach to the Data "
            "Protection Board within 72 hours. A detailed incident report "
            "shall be filed using the prescribed form.",
            "DPDP Act", "Tata Data Protection Policy"
        ),
        # Should be PARTIAL - covers breach but not 72 hours
        (
            "In the event of a personal data breach, the Data Fiduciary "
            "shall notify the Board within 72 hours of discovery.",
            "TMF shall report data breaches to relevant authorities. "
            "A breach response team will coordinate notification activities.",
            "DPDP Act", "Tata Data Protection Policy"
        ),
        # Should be NONE - completely different topic
        (
            "All SPs must implement KYC verification before onboarding customers.",
            "TMF shall use AES-256 encryption for all personal data at rest.",
            "AML Guidelines", "Tata Data Protection Policy"
        ),
    ]

    print("Coverage Checker Test (Heuristic Mode)")
    print("=" * 60)
    for reg_text, pol_text, reg_src, pol_src in tests:
        result = checker.check_coverage(
            reg_text, pol_text,
            regulation_source=reg_src,
            policy_source=pol_src
        )
        print(f"\nRegulation: {reg_text[:60]}...")
        print(f"Policy:     {pol_text[:60]}...")
        print(f"Coverage:   {result.coverage_type.value}")
        print(f"Score:      {result.compliance_score:.2f}")
        print(f"Missing:    {result.what_is_missing}")

    print(f"\nStats: {checker.get_stats()}")