"""
Document Chunker — Regulation-aware semantic chunking (FIXED VERSION)
"""

import re
import uuid
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  GOVERNANCE patterns — always reject these                           #
# ------------------------------------------------------------------ #
_GOVERNANCE_PATTERNS = [
    r'\bchairperson\b',
    r'\btribunal\b',
    r'\bcourt.*jurisdiction\b',
    r'\bterm of office\b',
    r'\bappointed by the (central|state) government\b',
    r'\bheadquarters.*shall be\b',
    r'\bno civil court\b',
    r'\bappellate\b',
]

# ------------------------------------------------------------------ #
#  DESCRIPTIVE patterns — reject if no obligation keyword present      #
# ------------------------------------------------------------------ #
_DESCRIPTIVE_PATTERNS = [
    r'^\s*(purpose|scope|background|introduction|applicability|definitions?)',
    r'\bshall be called\b',
    r'\bshall be known as\b',
    r'\bmeans and includes\b',
]

# ------------------------------------------------------------------ #
#  SECTION splitting                                                   #
# ------------------------------------------------------------------ #
_SECTION_BOUNDARY = re.compile(
    r'(?=\n{1,2}[ \t]*(?:'
    r'\d{1,2}\.\d{0,2}\.?\s+[A-Z]'
    r'|Article\s+\d+'
    r'|PART\s+[IVXLC\d]+'
    r'|Section\s+\d+'
    r'|Clause\s+\d+'
    r'|\(\d+\)\s+[A-Z]'
    r'))',
    re.MULTILINE
)

_HEADER_RE = re.compile(
    r'^('
    r'\d{1,2}(?:\.\d{1,2})*\.?\s+[^\n]{3,80}'
    r'|Article\s+\d+[^\n]{0,60}'
    r'|PART\s+[IVXLC\d]+[^\n]{0,60}'
    r'|Section\s+\d+[^\n]{0,60}'
    r'|Clause\s+\d+[^\n]{0,60}'
    r'|\(\d+\)\s+[A-Z][^\n]{5,60}'
    r')',
    re.IGNORECASE
)


# ------------------------------------------------------------------ #
#  TEXT NORMALISATION                                                  #
# ------------------------------------------------------------------ #

def _normalise(text: str) -> str:
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _heal_page_boundaries(full_text: str) -> str:
    def _replacer(m):
        before = m.group(1).rstrip()
        after  = m.group(2).lstrip()
        if not re.search(r'[.!?]$', before):
            return before + ' ' + after
        return m.group(0)
    return re.sub(r'([^\n]+)\n\n([^\n]+)', _replacer, full_text)


# ------------------------------------------------------------------ #
#  SINGLE UNIFIED VALIDITY FILTER                                      #
# ------------------------------------------------------------------ #

def _is_valid(text: str) -> bool:
    """
    Single unified filter. Keeps a chunk if:
    1. Long enough to be meaningful (10+ words)
    2. Starts with a capital letter or number
    3. Not a governance/administrative clause
    4. Either has an obligation keyword OR is long enough to be substantive

    Replaces the previous double-filter (is_real_obligation + _is_valid)
    which was too aggressive and rejected ~95% of valid regulation text.
    """
    stripped = text.strip()

    # Too short to be meaningful
    if len(stripped.split()) < 8:
        return False

    # Must start properly — capital letter, digit, or bracket
    if not re.match(r'^[A-Z0-9(\[]', stripped):
        return False
    
    # ── Extra noise filters (add after the capital-letter check) ──────────

# Reject fragments starting mid-sentence (year fragments like "1961), Groups...")
    if re.match(r'^\d{4}\)', stripped):
        return False

# Reject "Explanation:" definition lines
    if re.match(r'^explanation[\s:\-]', stripped, re.IGNORECASE):
        return False

# Reject effective/commencement date clauses
    if re.match(r'^(these guidelines|this (act|direction|circular|regulation))\s+shall\s+take\s+effect',
            stripped, re.IGNORECASE):
        return False
    if re.match(r'^effective date', stripped, re.IGNORECASE):
        return False

# Reject parliamentary laying clauses ("Every rule made...shall be laid before")
    if re.search(r'every (rule|notification|regulation).*shall be laid|laid before.*house|laid on the table', stripped, re.IGNORECASE):
        return False

# Reject Board procedure/governance clauses (supplements _GOVERNANCE_PATTERNS)
    if re.search(r'\bthe board shall observe\b|\bboard.*procedure.*meeting\b',
             stripped, re.IGNORECASE):
        return False

# Reject "No order...shall be made" procedural sub-clauses
    if re.match(r'^\(\d+\)\s+no order', stripped, re.IGNORECASE):
        return False

# Reject enclosure/cover letter lines
    if re.match(r'^(enclosure|madam|dear sir)', stripped, re.IGNORECASE):
        return False

    lower = stripped.lower()

    # Always reject governance/administrative clauses
    for pattern in _GOVERNANCE_PATTERNS:
        if re.search(pattern, lower):
            return False

    # Has obligation keyword — likely a valid regulatory requirement
    has_obligation = bool(re.search(
        r'\b(shall|must|required to|mandatorily)\b', lower
    ))

    if has_obligation:
        # Reject pure descriptive/naming statements even if they use "shall"
        for pattern in _DESCRIPTIVE_PATTERNS:
            if re.search(pattern, lower):
                return False
        return True

    # No obligation keyword — only keep if substantive (30+ words)
    # Captures context paragraphs that support obligation interpretation
    if len(stripped.split()) >= 30:
        return True

    return False


# ------------------------------------------------------------------ #
#  SENTENCE SPLITTING                                                  #
# ------------------------------------------------------------------ #

_SENTENCE_END = re.compile(r'(?<=[.!?;])\s+(?=[A-Z(])')


def _split_at_sentences(text: str, max_size: int) -> List[str]:
    sentences = _SENTENCE_END.split(text)
    chunks    = []
    current   = []

    for s in sentences:
        if len(" ".join(current + [s])) > max_size:
            chunks.append(" ".join(current))
            current = [s]
        else:
            current.append(s)

    if current:
        chunks.append(" ".join(current))

    return chunks


# ------------------------------------------------------------------ #
#  HEADER UTILITIES                                                    #
# ------------------------------------------------------------------ #

def _extract_header(text: str) -> str:
    match = _HEADER_RE.match(text.strip())
    return match.group(1) if match else ""


def _get_parent_section(header: str) -> str:
    m = re.match(r'^(\d+\.\d+)\.\d+', header)
    return m.group(1) if m else ""


# ------------------------------------------------------------------ #
#  DOCUMENT CHUNKER                                                    #
# ------------------------------------------------------------------ #

class DocumentChunker:

    def __init__(self, chunk_size: int = 600, document_type: str = "unknown"):
        self.chunk_size    = chunk_size
        self.document_type = document_type

    def chunk_document(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not pages:
            return []

        source = pages[0]['document_metadata']['file_name']

        # Build full text from all pages
        full_text = "\n\n".join(p['content'] for p in pages)
        full_text = _normalise(full_text)
        full_text = _heal_page_boundaries(full_text)

        # Split into sections at structural boundaries
        sections = _SECTION_BOUNDARY.split(full_text)

        chunks = []
        idx    = 0

        for section in sections:

            # Split multi-obligation content into sub-sections
            sub_sections = re.split(r'\n\n|•|;', section)

            for sub in sub_sections:

                # Split long sub-sections at sentence boundaries
                sentences = _split_at_sentences(sub, self.chunk_size)

                for chunk_text in sentences:

                    chunk_text = chunk_text.strip()

                    # Single unified validity filter
                    if not _is_valid(chunk_text):
                        continue

                    # Tag domain
                    from src.ingestion.domain_tagger import tag_domain
                    domain_result = tag_domain(chunk_text)

                    chunks.append({
                        "chunk_id":         str(uuid.uuid4()),
                        "content":          chunk_text,
                        "source":           source,
                        "doc_type":         self.document_type,
                        "chunk_index":      idx,
                        "domain":           domain_result.primary,
                        "domain_confidence": domain_result.confidence
                    })

                    idx += 1

        logger.info(f"{source}: {len(chunks)} clean regulation chunks")
        return chunks