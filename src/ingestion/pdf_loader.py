"""
PDF Loader — Clean text extraction using PyMuPDF.

Key fix over previous version:
  PyMuPDF's page.get_text() inserts a hard newline wherever the PDF
  renderer wrapped a line — even mid-sentence.  We now use get_text("blocks")
  which groups text by paragraph block, then re-join lines within each block
  so sentences are never split across lines before the chunker sees them.
"""

import re
import fitz                        # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


# Lines that are clearly page-furniture, not regulation text.
# Checked against each extracted block before it reaches the chunker.
_JUNK_PATTERNS = re.compile(
    r'^('
    r'page\s+\d+\s*(of\s*\d+)?'          # "Page 3 of 45"
    r'|\d+\s*/\s*\d+'                      # "3 / 45"
    r'|reserve\s+bank\s+of\s+india'        # RBI header line
    r'|master\s+direction'                 # running header
    r'|ministry\s+of'                      # govt header
    r'|government\s+of\s+india'
    r'|the\s+gazette\s+of\s+india'
    r'|official\s+gazette'
    r'|published\s+in\s+the\s+official'
    r'|printed\s+at'
    r'|www\.'                              # URL lines
    r'|https?://'
    r'|\d+$'                               # bare page number
    r')',
    re.IGNORECASE
)


def _clean_line(line: str) -> str:
    """
    Normalise a single line coming out of PyMuPDF:
      - strip surrounding whitespace
      - collapse multiple internal spaces / tabs
      - remove soft-hyphen line-break artifacts (word- \\n word → word word)
    """
    line = line.strip()
    line = re.sub(r'\s+', ' ', line)
    # Heal hyphenated line-breaks: "authen-\ntication" → "authentication"
    line = re.sub(r'-\s*$', '', line)
    return line


def _is_junk_line(line: str) -> bool:
    return bool(_JUNK_PATTERNS.match(line.strip().lower()))


def _join_block_lines(raw_block_text: str) -> str:
    """
    Given the raw text of one PyMuPDF block (which has hard \\n at every
    rendered line-wrap), produce clean prose:

    Rules:
    1. If a line ends with a sentence-terminating punctuation (. ! ? : ;)
       or is an enumeration item (a) b) i. ii.) → keep a real newline after it
       so the chunker can later split on paragraph boundaries.
    2. Otherwise → the line-break is just PDF word-wrap → replace with a space.
    3. Strip junk lines entirely.
    """
    lines = raw_block_text.splitlines()
    cleaned: List[str] = []

    for raw in lines:
        line = _clean_line(raw)
        if not line:
            continue
        if _is_junk_line(line):
            continue
        cleaned.append(line)

    if not cleaned:
        return ""

    result_parts: List[str] = []
    for i, line in enumerate(cleaned):
        result_parts.append(line)
        if i < len(cleaned) - 1:
            # Decide: real paragraph break or just PDF word-wrap?
            ends_sentence   = bool(re.search(r'[.!?;:]\s*$', line))
            next_is_header  = bool(re.match(
                r'^(\d{1,2}\.(\d{1,2})?\.?\s+[A-Z]|Article\s+\d|PART\s+[IVX\d]|Section\s+\d|\([a-z]\)|\([ivx]+\))',
                cleaned[i + 1]
            ))
            is_enum_item    = bool(re.match(r'^\([a-z]\)|^[a-z]\)', line))
            if ends_sentence or next_is_header or is_enum_item:
                result_parts.append('\n')
            else:
                result_parts.append(' ')

    return ''.join(result_parts).strip()


class PDFLoader:
    def __init__(self):
        self.supported_extensions = {'.pdf'}

    def load(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load a PDF and return a list of page dicts.

        Each dict has:
          content             : clean prose text (sentences not broken mid-line)
          page_number         : 1-based
          total_pages         : int
          width, height       : page dimensions in pts
          document_metadata   : title, author, total_pages, file_name
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {filepath}")
        if path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"Not a PDF: {path.suffix}")

        logger.info(f"Loading: {path.name}")
        documents = []

        with fitz.open(filepath) as doc:
            if doc.is_encrypted:
                if not doc.authenticate(""):
                    raise ValueError(f"PDF is password-protected: {path.name}")

            doc_metadata = {
                'title':       doc.metadata.get('title', ''),
                'author':      doc.metadata.get('author', ''),
                'total_pages': len(doc),
                'file_name':   path.name,
            }

            for page_num in range(len(doc)):
                page   = doc[page_num]
                blocks = page.get_text("blocks")   # list of (x0,y0,x1,y1,text,block_no,block_type)

                # Sort top-to-bottom, left-to-right (handles two-column layouts)
                blocks = sorted(blocks, key=lambda b: (round(b[1] / 20), b[0]))

                page_paragraphs: List[str] = []
                for block in blocks:
                    block_type = block[6]
                    if block_type != 0:          # 0 = text, 1 = image
                        continue
                    raw_text = block[4]
                    clean = _join_block_lines(raw_text)
                    if clean:
                        page_paragraphs.append(clean)

                page_text = '\n\n'.join(page_paragraphs)

                if not page_text.strip():
                    logger.debug(f"Page {page_num + 1} is empty after cleaning — skipped")
                    continue

                documents.append({
                    'content':           page_text,
                    'page_number':       page_num + 1,
                    'total_pages':       len(doc),
                    'width':             page.rect.width,
                    'height':            page.rect.height,
                    'document_metadata': doc_metadata,
                })

        logger.info(f"Extracted {len(documents)} pages from {path.name}")
        return documents