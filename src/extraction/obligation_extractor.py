"""
Obligation Extractor - Classifies chunks and extracts structured obligations.

This is the most important new component. It answers:
1. Is this chunk even an obligation? (or definition/example/exemption?)
2. If yes, WHAT EXACTLY is required?

This uses Groq to extract structured obligations, which then get
checked against policy text in the CoverageChecker.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Optional, Tuple
import json
import logging
import time
import re
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── CHUNK TYPE CLASSIFIER (no LLM needed, fast regex) ─────────────────────

# Chunks matching these patterns are NOT obligations
# They are definitions, examples, illustrations, or exemptions
SKIP_PATTERNS = [
    # Definitions
    r'^\s*[\(\w]+\s*["\']?\w+["\']?\s+means\s+',
    r'^\s*"[\w\s]+" means ',
    r'^\s*[\(\w]+\s*"interpretation"',
    r'^\s*for the (purpose|purposes) of this (act|section|rule)',
    # Examples / Illustrations
    r'^\s*(illustration|example)\.',
    r'^X, an individual',
    r'^Y, a (company|person|entity)',
    # Index / Table of Contents
    r'^\s*(index|sl\. no|contents|page no)',
    r'^\s*\d+\.\s+(introduction|section|annexure)',
    # Whereas / Preamble
    r'^\s*whereas,?\s+the\s+(central government|reserve bank)',
    # Pure cross references
    r'^\s*as (prescribed|mentioned|referred to) (under|in) (rule|section|article)',
    # ── NEW: DPDP-specific non-obligations ────────────────────────────────
    # Lettered definitions like (a) "Consent Manager" means
    r'^\([a-z]\)\s+"[A-Z][a-zA-Z\s]+" means',
    # And whereas preamble clauses
    r'^And whereas',
    # Explanation clauses
    r'^Explanation\.?\s*[—\-]',
    # Illustration clauses
    r'^Illustration\.?\s*[—\-]',
    # Government power/notification clauses
    r'Central Government may.*notify',
    # Penalty collection clauses
    r'sums realised by way of penalties',
    # Annexure headers
    r'^Annexure [IVX]+',
    # Commencement / extent clauses
    r'It shall come into force on such date',
    r'^\s*\(\d+\)\s+It shall come into force',
    # Board structure / establishment clauses
    r'The Board shall be a body corporate',
    r'^\s*\(\d+\)\s+The Board shall be a body corporate',
    # Lettered definitions like (g) "Consent Manager" means
    r'^\s*\([a-z]\)\s+"[A-Z]',
    # Any quoted term followed by means
    r'"[A-Za-z\s]+" means ',
    # Board procedural clauses (not company obligations)
    r'^\s*The Board (shall|may) (function|determine|conduct|accept|require)',
    # X as any entity illustration
    r'^X, a (telecom|company|bank|person|fintech)',
]

# Chunks containing these words are likely real obligations
OBLIGATION_INDICATORS = [
    'shall', 'must', 'required to', 'obligated', 'mandatory',
    'prohibited', 'shall not', 'must not', 'is required',
    'have to', 'needs to', 'are required'
]


EXTRACTION_PROMPT = """You are a regulatory compliance expert analyzing Indian fintech regulations.

Extract ALL compliance obligations from this regulation text.

REGULATION TEXT:
{regulation_text}

SOURCE: {source}

Instructions:
- Extract only MANDATORY obligations (shall, must, required)
- Ignore definitions, examples, cross-references
- Be specific about timeframes, thresholds, and technical requirements
- Each obligation should be one specific, checkable requirement

Respond ONLY with this JSON (no other text):
{{
    "chunk_type": "obligation|definition|example|procedure|exemption|mixed",
    "has_obligations": true,
    "obligations": [
        {{
            "id": "obl_001",
            "subject": "who must comply (e.g. Data Fiduciary, NBFC, SP, RE)",
            "action": "what must be done (single verb phrase)",
            "object": "what it applies to",
            "specifics": "any timeframes/thresholds/methods (null if none)",
            "mandatory": true,
            "verbatim_phrase": "the exact shall/must phrase from text"
        }}
    ],
    "skip_reason": "null or reason why chunk should be skipped"
}}"""


class ObligationExtractor:
    """
    Classifies regulation chunks and extracts structured obligations.
    
    Two-stage process:
    1. Fast regex pre-filter (no API call needed for obvious definitions)
    2. LLM extraction for obligation chunks
    """

    def __init__(self, groq_client=None):
        """
        Args:
            groq_client: Your existing GroqLLMClient instance.
                         Pass None to use regex-only mode (no LLM).
        """
        self.groq_client = groq_client
        self.stats = {
            'total': 0,
            'skipped_regex': 0,
            'obligations_found': 0,
            'definitions_filtered': 0,
            'llm_calls': 0
        }

    def is_definition_by_regex(self, text: str) -> Tuple[bool, str]:
        """
        Fast regex check - no API call needed.
        Returns (is_definition, reason)
        """
        text_lower = text.lower().strip()

        # Normalize curly/smart quotes to straight quotes before matching
        text = text.replace('\u201c', '"').replace('\u201d', '"')
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        text = text.replace('\u2014', '-').replace('\u2013', '-')  # em/en dash

        for pattern in SKIP_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE):
                return True, f"Matches skip pattern: {pattern[:40]}"

        # Check if any obligation indicators exist
        has_obligation = any(ind in text_lower for ind in OBLIGATION_INDICATORS)
        if not has_obligation:
            # Short chunks with no obligation language are likely definitions
            if len(text.split()) < 30:
                return True, "Short chunk with no obligation language"

        return False, ""

    def extract_obligations_with_llm(
        self,
        regulation_text: str,
        source: str = ""
    ) -> Dict:
        """
        Use Groq LLM to extract structured obligations.
        Falls back to basic extraction if LLM unavailable.
        """
        if self.groq_client is None:
            return self._regex_fallback_extraction(regulation_text)

        try:
            prompt = EXTRACTION_PROMPT.format(
                regulation_text=regulation_text[:1500],
                source=source
            )

            response = self.groq_client.generate(
                prompt=prompt,
                system_prompt=(
                    "You are a regulatory compliance expert. "
                    "Extract obligations precisely. "
                    "Respond with valid JSON only."
                ),
                temperature=0.0,
                max_tokens=800
            )

            self.stats['llm_calls'] += 1

            # Parse response
            text = response.text.strip()
            # Strip markdown if present
            text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=re.MULTILINE)

            result = json.loads(text)
            return result

        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}, using fallback")
            return self._regex_fallback_extraction(regulation_text)

    def _regex_fallback_extraction(self, text: str) -> Dict:
        """
        Fallback when LLM is unavailable.
        Less accurate but still useful.
        """
        obligations = []
        sentences = re.split(r'(?<=[.;])\s+', text)

        for i, sentence in enumerate(sentences):
            if any(ind in sentence.lower() for ind in OBLIGATION_INDICATORS):
                # Try to extract subject
                subject = "Entity"
                for subj in ["Data Fiduciary", "NBFC", "SP", "RE",
                             "Data Principal", "Every reporting entity",
                             "Every intermediary"]:
                    if subj.lower() in sentence.lower():
                        subject = subj
                        break

                obligations.append({
                    "id": f"obl_{i:03d}",
                    "subject": subject,
                    "action": sentence[:100],
                    "object": "as specified",
                    "specifics": None,
                    "mandatory": True,
                    "verbatim_phrase": sentence[:200]
                })

        return {
            "chunk_type": "obligation" if obligations else "unknown",
            "has_obligations": len(obligations) > 0,
            "obligations": obligations,
            "skip_reason": None
        }

    def process_chunk(
        self,
        regulation_text: str,
        regulation_source: str = "",
        use_llm: bool = True
    ) -> Dict:
        """
        Main entry point. Process a single regulation chunk.
        
        Returns:
            {
                "should_skip": bool,
                "skip_reason": str or None,
                "chunk_type": str,
                "obligations": list,
                "has_obligations": bool
            }
        """
        self.stats['total'] += 1

        # Stage 1: Fast regex filter
        is_def, reason = self.is_definition_by_regex(regulation_text)
        if is_def:
            self.stats['skipped_regex'] += 1
            self.stats['definitions_filtered'] += 1
            return {
                "should_skip": True,
                "skip_reason": reason,
                "chunk_type": "definition",
                "obligations": [],
                "has_obligations": False
            }

        # Stage 2: LLM extraction (or regex fallback)
        if use_llm and self.groq_client is not None:
            extraction = self.extract_obligations_with_llm(
                regulation_text, regulation_source
            )
        else:
            extraction = self._regex_fallback_extraction(regulation_text)

        chunk_type = extraction.get("chunk_type", "unknown")
        obligations = extraction.get("obligations", [])
        skip_reason = extraction.get("skip_reason")

        # Skip if LLM says it's a definition/example
        if chunk_type in ["definition", "example", "exemption"]:
            self.stats['definitions_filtered'] += 1
            return {
                "should_skip": True,
                "skip_reason": skip_reason or f"LLM classified as {chunk_type}",
                "chunk_type": chunk_type,
                "obligations": [],
                "has_obligations": False
            }

        if obligations:
            self.stats['obligations_found'] += len(obligations)

        return {
            "should_skip": False,
            "skip_reason": None,
            "chunk_type": chunk_type,
            "obligations": obligations,
            "has_obligations": len(obligations) > 0
        }

    def get_stats(self) -> Dict:
        total = self.stats['total']
        return {
            **self.stats,
            'skip_rate': round(
                self.stats['skipped_regex'] / total * 100
                if total > 0 else 0, 1
            )
        }


# ── TEST ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    extractor = ObligationExtractor(groq_client=None)  # regex mode

    test_chunks = [
        # Should SKIP - definition
        ('(r) "notification" means a notification published in the Official Gazette.',
         "dpdp_act.pdf"),
        # Should SKIP - illustration
        ('X, an individual, while blogging her views, has published her name.',
         "dpdp_act.pdf"),
        # Should PROCESS - real obligation
        ('In the event of a personal data breach, the Data Fiduciary shall '
         'notify the Board and each affected Data Principal of such breach '
         'in such manner and within such time as may be prescribed.',
         "dpdp_act.pdf"),
        # Should PROCESS - real obligation
        ('All SPs are required to implement and maintain an AML/CFT/CPF program '
         'that includes customer due diligence, transaction monitoring, and '
         'suspicious transaction reporting.',
         "aml_guidelines.pdf"),
        # Should SKIP - table of contents
        ('INDEX Sl. No Contents Page No. Introduction Section 1',
         "nbfc_it.pdf"),
    ]

    print("Obligation Extractor Test (Regex Mode)")
    print("=" * 60)
    for text, source in test_chunks:
        result = extractor.process_chunk(text, source, use_llm=False)
        status = "SKIP" if result['should_skip'] else "PROCESS"
        print(f"\n[{status}] {text[:70]}...")
        if result['should_skip']:
            print(f"  Reason: {result['skip_reason']}")
        else:
            print(f"  Type: {result['chunk_type']}")
            print(f"  Obligations found: {len(result['obligations'])}")
            for obl in result['obligations'][:2]:
                print(f"  - {obl['action'][:60]}...")

    print(f"\nStats: {extractor.get_stats()}")