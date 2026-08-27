"""
Regulatory Change Monitor — src/change_monitor.py
REG-INSIGHT v2 | Zil Money Internship Project

Detects new, modified, removed, and unchanged obligations
when a new version of a regulation PDF is uploaded.
"""

import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any

# ── Thresholds ───────────────────────────────────────────────────────────────
UNCHANGED_THRESHOLD = 0.90
MODIFIED_THRESHOLD  = 0.60

# ── Shared components (lazy-loaded once) ─────────────────────────────────────
_model    = None
_splitter = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _get_splitter() -> RecursiveCharacterTextSplitter:
    global _splitter
    if _splitter is None:
        _splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
        )
    return _splitter


# ── Core helpers ──────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = [page.get_text() for page in doc]
    doc.close()
    return " ".join(pages)


def chunk_text(text: str) -> List[str]:
    return _get_splitter().split_text(text)


# ── Main analysis function ────────────────────────────────────────────────────

def analyse_changes(
    new_pdf_bytes: bytes,
    regulation_name: str,
    chroma_collection: Any,
) -> Dict[str, Any]:

    empty_result: Dict[str, Any] = {
        "new": [],
        "modified": [],
        "removed": [],
        "unchanged_count": 0,
        "total_new_chunks": 0,
        "total_old_chunks": 0,
        "error": None,
    }

    # ── Step 1: Extract & chunk new PDF ──────────────────────────────────────
    try:
        raw_text = extract_text_from_pdf(new_pdf_bytes)
    except Exception as exc:
        empty_result["error"] = f"PDF extraction failed: {exc}"
        return empty_result

    new_chunks = chunk_text(raw_text)
    if not new_chunks:
        empty_result["error"] = "No text could be extracted from the uploaded PDF."
        return empty_result

    # ── Step 2: Embed new chunks ──────────────────────────────────────────────
    model = _get_model()
    new_embeddings: np.ndarray = model.encode(
        new_chunks, normalize_embeddings=True, show_progress_bar=False
    )

    # ── Step 3: Retrieve old chunks from ChromaDB ─────────────────────────────
    try:
        old_results = chroma_collection.get(
            where={"regulation_name": regulation_name},
            include=["documents", "embeddings"],
        )
    except Exception as exc:
        empty_result["error"] = f"ChromaDB query failed: {exc}"
        return empty_result

    old_chunks: List[str] = old_results.get("documents") or []

    # ── Safe numpy construction — never use `or` / `if` on numpy arrays ───────
    raw_old_embs = old_results.get("embeddings")

    # Check emptiness without triggering numpy's ambiguous truth value error
    if raw_old_embs is None:
        has_old_embeddings = False
    elif isinstance(raw_old_embs, np.ndarray):
        has_old_embeddings = raw_old_embs.size > 0
    else:
        has_old_embeddings = len(raw_old_embs) > 0

    if has_old_embeddings:
        old_embeddings = np.array(raw_old_embs, dtype=np.float32)
    else:
        old_embeddings = np.empty((0, new_embeddings.shape[1]), dtype=np.float32)

    # ── Step 4: Classify each new chunk ──────────────────────────────────────
    new_obligations:       List[str]            = []
    modified_obligations:  List[Dict[str, Any]] = []
    unchanged_obligations: List[str]            = []
    matched_old_indices:   set                  = set()

    no_old_data = old_embeddings.shape[0] == 0

    for new_chunk, new_emb in zip(new_chunks, new_embeddings):
        if no_old_data:
            new_obligations.append(new_chunk)
            continue

        similarities: np.ndarray = old_embeddings @ new_emb
        best_idx:   int   = int(np.argmax(similarities))
        best_score: float = float(similarities[best_idx])

        if best_score >= UNCHANGED_THRESHOLD:
            unchanged_obligations.append(new_chunk)
            matched_old_indices.add(best_idx)
        elif best_score >= MODIFIED_THRESHOLD:
            modified_obligations.append({
                "new":        new_chunk,
                "old":        old_chunks[best_idx],
                "similarity": round(best_score, 4),
            })
            matched_old_indices.add(best_idx)
        else:
            new_obligations.append(new_chunk)

    # ── Step 5: Find removed obligations ─────────────────────────────────────
    removed_obligations: List[str] = [
        old_chunks[i]
        for i in range(len(old_chunks))
        if i not in matched_old_indices
    ]

    return {
        "new":              new_obligations,
        "modified":         modified_obligations,
        "removed":          removed_obligations,
        "unchanged_count":  len(unchanged_obligations),
        "total_new_chunks": len(new_chunks),
        "total_old_chunks": len(old_chunks),
        "error":            None,
    }