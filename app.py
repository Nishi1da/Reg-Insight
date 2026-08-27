"""
REG-INSIGHT — Regulatory Compliance Gap Detection System
Streamlit UI — connects to run_compliance_analysis.py output
"""

import streamlit as st
import json
import time
import io
import csv
import re
import sys
from pathlib import Path
from datetime import datetime

from detail_view import WEEK8_CSS, page_explorer_v2
from export import page_export_v2
from src.change_monitor import analyse_changes
import requests

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="REG-INSIGHT",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

:root {
    --navy:    #0D1B2A;
    --ink:     #1A2B3C;
    --teal:    #0E7C7B;
    --teal-lt: #E6F4F4;
    --amber:   #D4820A;
    --amber-lt:#FDF3E0;
    --red:     #C0392B;
    --red-lt:  #FDECEA;
    --green:   #1E7E55;
    --green-lt:#E8F5EE;
    --slate:   #64748B;
    --border:  #E2E8F0;
    --bg:      #F8FAFC;
}

[data-testid="stSidebar"] { background: var(--navy); border-right: none; }
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] hr { border-color: #1E3A5F; margin: 1rem 0; }

.page-header {
    background: linear-gradient(135deg, var(--navy) 0%, #1A3A5C 100%);
    color: white; padding: 2rem 2.5rem; border-radius: 12px;
    margin-bottom: 1.5rem; position: relative; overflow: hidden;
}
.page-header::after {
    content: ''; position: absolute; top: -40px; right: -40px;
    width: 180px; height: 180px; background: rgba(14,124,123,0.15); border-radius: 50%;
}
.page-header h1 {
    font-family: 'DM Serif Display', serif; font-size: 2rem;
    margin: 0 0 0.3rem; letter-spacing: -0.02em;
}
.page-header p { margin: 0; color: #94A3B8; font-size: 0.9rem; }

.metric-card {
    background: white; border: 1px solid var(--border);
    border-radius: 10px; padding: 1.25rem 1.5rem;
    position: relative; overflow: hidden;
}
.metric-card .label {
    font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 0.08em; color: var(--slate); margin-bottom: 0.4rem;
}
.metric-card .value {
    font-family: 'DM Serif Display', serif; font-size: 2.2rem;
    line-height: 1; color: var(--navy);
}
.metric-card .sub { font-size: 0.78rem; color: var(--slate); margin-top: 0.25rem; }
.metric-card.green  { border-left: 4px solid var(--green); }
.metric-card.amber  { border-left: 4px solid var(--amber); }
.metric-card.red    { border-left: 4px solid var(--red); }
.metric-card.teal   { border-left: 4px solid var(--teal); }

.result-row {
    background: white; border: 1px solid var(--border);
    border-radius: 8px; padding: 0.9rem 1.1rem; margin-bottom: 0.5rem;
    transition: box-shadow 0.15s;
}
.result-row:hover { box-shadow: 0 2px 12px rgba(0,0,0,0.08); }
.result-row .reg-text { font-size: 0.85rem; color: var(--navy); font-weight: 500; margin-bottom: 0.15rem; }
.result-row .meta { font-size: 0.75rem; color: var(--slate); }

.detail-panel {
    background: white; border: 1px solid var(--border);
    border-radius: 10px; padding: 1.5rem;
}
.detail-panel .section-label {
    font-size: 0.7rem; text-transform: uppercase;
    letter-spacing: 0.1em; color: var(--slate); margin-bottom: 0.4rem;
}
.text-block {
    background: var(--bg); border-left: 3px solid var(--border);
    padding: 0.75rem 1rem; border-radius: 0 6px 6px 0;
    font-size: 0.83rem; line-height: 1.6; color: var(--ink); margin-bottom: 1rem;
}
.text-block.regulation    { border-color: var(--teal); }
.text-block.policy        { border-color: var(--amber); }
.text-block.gap-block     { border-color: var(--red); }
.text-block.covered-block { border-color: var(--green); }

.score-bar-wrap { margin: 0.3rem 0 0.8rem; }
.score-bar-bg { height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }
.score-bar-fill { height: 100%; border-radius: 3px; transition: width 0.4s ease; }

.explanation-card {
    background: linear-gradient(135deg, #F0F7FF 0%, #E8F4F8 100%);
    border: 1px solid #B8D9F0; border-radius: 10px; padding: 1.25rem 1.5rem;
}
.explanation-card .exp-title { font-weight: 600; color: var(--navy); font-size: 0.88rem; margin-bottom: 0.5rem; }
.explanation-card .exp-body  { color: var(--ink); font-size: 0.84rem; line-height: 1.65; }

.section-divider { border: none; border-top: 1px solid var(--border); margin: 1.25rem 0; }

.step-row { display: flex; align-items: center; gap: 0.75rem; padding: 0.6rem 0; font-size: 0.88rem; }
.step-dot {
    width: 28px; height: 28px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem; font-weight: 700; flex-shrink: 0;
}
.step-dot.done    { background: var(--green);  color: white; }
.step-dot.active  { background: var(--teal);   color: white; }
.step-dot.waiting { background: var(--border); color: var(--slate); }

.stButton > button {
    background: var(--teal) !important; color: white !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important;
    padding: 0.5rem 1.5rem !important;
}
.stButton > button:hover {
    background: #0A6665 !important;
    box-shadow: 0 4px 12px rgba(14,124,123,0.3) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Inject Week 8 CSS ─────────────────────────────────────────────────────────
st.markdown(WEEK8_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HTML STRIP UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def strip_html(text: str) -> str:
    """Remove HTML tags and decode common HTML entities from text."""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&nbsp;',  ' ',  text)
    text = re.sub(r'&amp;',   '&',  text)
    text = re.sub(r'&lt;',    '<',  text)
    text = re.sub(r'&gt;',    '>',  text)
    text = re.sub(r'&quot;',  '"',  text)
    text = re.sub(r'&#39;',   "'",  text)
    text = re.sub(r'\s+',     ' ',  text).strip()
    return text


# ══════════════════════════════════════════════════════════════════════════════
# FORMAT ADAPTER
# ══════════════════════════════════════════════════════════════════════════════

def normalize_item(item: dict) -> dict:
    out = dict(item)

    # 1. Regulation text
    out["regulation_text"] = strip_html(
        item.get("regulation_text") or item.get("chunk_text") or ""
    )

    # 2. Regulation source
    out["regulation_source"] = (
        item.get("regulation_source") or
        item.get("source") or
        item.get("regulation") or ""
    )

    # 3. Classification
    new_status = (item.get("compliance_status") or "").strip().lower()
    old_cls    = (item.get("classification") or "").strip().lower()

    if new_status:
        mapping = {
            "compliant":               "aligned",
            "substantially_compliant": "partial",
            "partially_compliant":     "partial",
            "non_compliant":           "gap",
            "not_applicable":          "unmatched",
        }
        out["classification"]    = mapping.get(new_status, "gap")
        out["compliance_status"] = new_status
    elif old_cls:
        valid = {"aligned", "partial", "gap", "unmatched"}
        out["classification"] = old_cls if old_cls in valid else "gap"
        rev = {
            "aligned":   "compliant",
            "partial":   "partially_compliant",
            "gap":       "non_compliant",
            "unmatched": "not_applicable",
        }
        out["compliance_status"] = rev.get(old_cls, "non_compliant")
    else:
        out["classification"]    = "gap"
        out["compliance_status"] = "non_compliant"

    # 4. Score
    out["final_score"]         = float(item.get("compliance_score") or item.get("final_score") or item.get("best_score") or 0.0)
    out["bi_encoder_score"]    = float(item.get("bi_encoder_score") or 0.0)
    out["cross_encoder_score"] = float(item.get("cross_encoder_score") or 0.0)

    # 5. Policy text/source
    pol_text    = strip_html(item.get("policy_text", ""))
    pol_source  = item.get("policy_document") or item.get("matched_policy", "")
    old_matches = item.get("policy_matches", []) or []
    if old_matches and isinstance(old_matches[0], dict):
        top        = old_matches[0]
        pol_text   = pol_text or strip_html(
            top.get("policy_text") or top.get("policy_chunk_preview", "")
        )
        pol_source = pol_source or top.get("policy_source", "")
    out["_pol_text"]   = pol_text
    out["_pol_source"] = pol_source
    if not old_matches:
        out["policy_matches"] = [{"policy_text": pol_text, "policy_source": pol_source}]

    # 6. LLM explanation
    old_exp = item.get("llm_explanation", {}) or {}
    if old_exp:
        out["llm_explanation"] = old_exp
    else:
        missing   = item.get("what_is_missing", "")
        reasoning = item.get("compliance_reasoning", "") or item.get("reasoning", "")
        cov_type  = item.get("coverage_type", "")
        cls       = out["classification"]
        risk      = "high" if cls == "gap" else ("medium" if cls == "partial" else "low")
        out["llm_explanation"] = {
            "summary":            reasoning or (f"Coverage type: {cov_type}." if cov_type else ""),
            "gap_description":    missing,
            "covered_by":         item.get("what_policy_covers", ""),
            "risk_level":         risk,
            "recommended_action": item.get("recommended_action", ""),
            "coverage_type":      cov_type,
        }

    # 7. Pass-through fields
    out["what_is_missing"]      = item.get("what_is_missing", "")
    out["what_policy_covers"]   = item.get("what_policy_covers", "")
    out["compliance_reasoning"] = item.get("compliance_reasoning", "")
    out["coverage_type"]        = item.get("coverage_type", "")
    out["obligations"]          = item.get("obligations", [])

    # 8. regulation_metadata
    out["regulation_metadata"] = item.get("regulation_metadata") or {
        "domain":         item.get("category", ""),
        "page_number":    item.get("page_number", ""),
        "section_header": item.get("section_header", ""),
    }

    return out


def load_and_normalize(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        st.error(f"Failed to read JSON: {e}")
        return None, None, None

    items_raw   = []
    _source_key = None

    if isinstance(raw, list):
        items_raw   = raw
        _source_key = "(root list)"
    elif isinstance(raw, dict):
        preferred = ["regulation_analysis", "results", "gaps", "items",
                     "analysis", "regulations", "compliance_results", "data", "records"]
        for key in preferred:
            if key in raw and isinstance(raw[key], list) and len(raw[key]) > 0:
                items_raw   = raw[key]
                _source_key = key
                break
        if not items_raw:
            best_key, best_len = None, 0
            for k, v in raw.items():
                if isinstance(v, list) and len(v) > best_len:
                    best_key, best_len = k, len(v)
            if best_key:
                items_raw   = raw[best_key]
                _source_key = best_key

    if not items_raw:
        st.error(
            f"Could not find a list of items in the JSON. "
            f"Top-level keys: {list(raw.keys()) if isinstance(raw, dict) else type(raw)}"
        )
        return None, raw, None

    normalized = [normalize_item(i) for i in items_raw]
    summary    = _build_summary(normalized, raw)
    summary["_source_key"]     = _source_key
    summary["_raw_item_count"] = len(items_raw)

    return normalized, raw, summary


def _build_summary(normalized_items, raw):
    raw_summary = raw.get("summary", {}) if isinstance(raw, dict) else {}
    total     = len(normalized_items)
    aligned   = sum(1 for i in normalized_items if i["classification"] == "aligned")
    partial   = sum(1 for i in normalized_items if i["classification"] == "partial")
    gap       = sum(1 for i in normalized_items if i["classification"] == "gap")
    unmatched = sum(1 for i in normalized_items if i["classification"] == "unmatched")
    coverage  = round(aligned / total * 100, 1) if total > 0 else 0
    return {
        "total": total, "aligned": aligned, "partial": partial,
        "gap": gap, "unmatched": unmatched, "coverage": coverage,
        "compliance_breakdown": raw_summary.get("compliance_breakdown", {}),
        "by_regulation":        raw_summary.get("by_regulation", {}),
        "report_version":       raw.get("report_version", "") if isinstance(raw, dict) else "",
        "generated_at":         raw.get("generated_at", "") if isinstance(raw, dict) else "",
    }


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "page":              "home",
        "analysis_done":     False,
        "reg_items":         None,
        "raw_report":        None,
        "report_data":       None,
        "gap_results":       None,
        "chroma_collection": None,
        "selected_regulation": None,
        "summary":           None,
        "selected_item_idx": None,
        "regulation_file":   None,
        "policy_file":       None,
        "explorer_page":     0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Auto-load ChromaDB on startup ────────────────────────────────
    if st.session_state.chroma_collection is None:
        try:
            import chromadb
            client = chromadb.PersistentClient(path="data/processed/chroma_db")
            st.session_state.chroma_collection = client.get_collection("regulations")
        except Exception as e:
             st.session_state.chroma_error = str(e)
init_state()

PRELOAD_PATHS = [
    "outputs/compliance_llm_full.json",
    "outputs/compliance_heuristic.json",
    "outputs/compliance_report_v2.json",
    "outputs/thesis_gap_analysis9_explained.json",
    "outputs/thesis_gap_analysis9_enriched.json",
    "outputs/thesis_gap_analysis9.json",
]


def find_preloaded():
    for p in PRELOAD_PATHS:
        if Path(p).exists():
            return p
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def badge_html(classification):
    cls    = (classification or "unmatched").lower()
    labels = {"aligned": "Aligned", "partial": "Partial", "gap": "Gap", "unmatched": "Unmatched"}
    styles = {
        "aligned":   "background:#E8F5EE;color:#1E7E55",
        "partial":   "background:#FDF3E0;color:#D4820A",
        "gap":       "background:#FDECEA;color:#C0392B",
        "unmatched": "background:#F1F5F9;color:#64748B",
    }
    label = labels.get(cls, cls.title())
    style = styles.get(cls, "background:#F1F5F9;color:#64748B")
    return (f'<span style="{style};display:inline-block;padding:0.2rem 0.65rem;'
            f'border-radius:20px;font-size:0.72rem;font-weight:600;'
            f'letter-spacing:0.04em;text-transform:uppercase">{label}</span>')


def risk_html(risk):
    r      = (risk or "").lower()
    styles = {
        "high":   "background:#FDECEA;color:#C0392B",
        "medium": "background:#FDF3E0;color:#D4820A",
        "low":    "background:#E8F5EE;color:#1E7E55",
    }
    style = styles.get(r, "background:#F1F5F9;color:#64748B")
    label = (risk or "unknown").title() + " Risk"
    return (f'<span style="{style};display:inline-block;padding:0.2rem 0.65rem;'
            f'border-radius:20px;font-size:0.72rem;font-weight:600;'
            f'letter-spacing:0.04em;text-transform:uppercase">{label}</span>')


def score_bar(score, color="#0E7C7B", label="Match Score"):
    pct = int(min(max((score or 0), 0), 1) * 100)
    return (f'<div class="score-bar-wrap">'
            f'<div style="display:flex;justify-content:space-between;'
            f'font-size:0.75rem;color:#64748B;margin-bottom:3px">'
            f'<span>{label}</span><span>{score:.2f}</span></div>'
            f'<div class="score-bar-bg">'
            f'<div class="score-bar-fill" style="width:{pct}%;background:{color}"></div>'
            f'</div></div>')


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div style="font-family:\'DM Serif Display\',serif;font-size:1.4rem;'
            'color:white;letter-spacing:-0.02em;padding:0.5rem 0 0.25rem">'
            'REG-INSIGHT</div>'
            '<div style="font-size:0.72rem;color:#94A3B8;text-transform:uppercase;'
            'letter-spacing:0.08em;margin-bottom:0.75rem">FinTech Compliance</div>',
            unsafe_allow_html=True
        )
        st.markdown("<hr>", unsafe_allow_html=True)

        pages = {
            "home":           ("🏠", "Home"),
            "analyze":        ("⚙️", "Run Analysis"),
            "results":        ("📊", "Results"),
            "explorer":       ("🔍", "Gap Explorer"),
            "export":         ("📥", "Export"),
            "change_monitor": ("🔄", "Change Monitor"),   # ← NEW
        }
        for key, (icon, label) in pages.items():
            if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True):
                st.session_state.page = key
                st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)

        if st.session_state.analysis_done and st.session_state.reg_items:
            _live      = st.session_state.reg_items
            _total     = len(_live)
            _aligned   = sum(1 for i in _live if i["classification"] == "aligned")
            _partial   = sum(1 for i in _live if i["classification"] == "partial")
            _gap       = sum(1 for i in _live if i["classification"] == "gap")
            _unmatched = sum(1 for i in _live if i["classification"] == "unmatched")
            _coverage  = round(_aligned / _total * 100, 1) if _total > 0 else 0
            s          = st.session_state.summary or {}
            st.markdown(
                f'<div style="font-size:0.75rem;color:#94A3B8;margin-bottom:0.5rem;'
                f'text-transform:uppercase;letter-spacing:0.08em">Current Analysis</div>'
                f'<div style="color:white;font-size:0.85rem;line-height:1.8">'
                f'📋 {_total} regulations<br>'
                f'✅ {_aligned} aligned ({_coverage}%)<br>'
                f'🟡 {_partial} partial<br>'
                f'🔴 {_gap} gaps<br>'
                f'⬜ {_unmatched} unmatched</div>',
                unsafe_allow_html=True
            )
            if s.get("report_version"):
                st.markdown(
                    f'<div style="font-size:0.68rem;color:#475569;margin-top:0.5rem">'
                    f'v: {s["report_version"]}</div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                '<div style="font-size:0.78rem;color:#64748B;font-style:italic">'
                'No analysis loaded yet.<br>Go to Run Analysis to start.</div>',
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════

def page_home():
    st.markdown("""<div class="page-header">
        <h1>REG-INSIGHT</h1>
        <p>Automated Regulatory Compliance Gap Detection for Indian FinTech Companies</p>
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    cards = [
        ("teal",  "What It Does",
         "Compares Indian financial regulation PDFs against company policy documents "
         "to automatically identify compliance gaps using semantic search + LLM verification."),
        ("amber", "Who It's For",
         "Compliance officers who need to audit policy coverage without "
         "reading hundreds of pages manually."),
        ("green", "How It Works",
         "Uses semantic search + cross-encoder scoring + LLM explanations "
         "to classify each regulation as aligned, partial, gap, or unmatched."),
    ]
    for col, (color, label, text) in zip([col1, col2, col3], cards):
        with col:
            st.markdown(
                f'<div class="metric-card {color}">'
                f'<div class="label">{label}</div>'
                f'<div style="font-size:0.88rem;color:#1A2B3C;line-height:1.7;margin-top:0.5rem">'
                f'{text}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 🚀 Quick Start")
        preload_path = find_preloaded()
        if preload_path:
            st.success(f"Analysis found: `{Path(preload_path).name}`")
            if st.button("Load Results →", use_container_width=True):
                normalized, raw, summary = load_and_normalize(preload_path)
                if normalized:
                    st.session_state.reg_items     = normalized
                    st.session_state.raw_report    = raw
                    st.session_state.report_data   = raw
                    st.session_state.summary       = summary
                    st.session_state.analysis_done = True
                    st.session_state.page          = "results"
                    st.rerun()
        else:
            st.info("No pre-computed analysis found. Use Run Analysis to generate one.")
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        if st.button("⚙️ Run New Analysis", use_container_width=True):
            st.session_state.page = "analyze"
            st.rerun()

    with col_b:
        st.markdown("#### 📐 Classification Guide")
        st.markdown("""<div style="font-size:0.84rem;line-height:2.2">
            <span style="background:#E8F5EE;color:#1E7E55;padding:2px 10px;
                border-radius:20px;font-size:0.72rem;font-weight:600">ALIGNED</span>
            &nbsp; Policy explicitly covers the requirement<br>
            <span style="background:#FDF3E0;color:#D4820A;padding:2px 10px;
                border-radius:20px;font-size:0.72rem;font-weight:600">PARTIAL</span>
            &nbsp; Policy addresses topic but misses specifics<br>
            <span style="background:#FDECEA;color:#C0392B;padding:2px 10px;
                border-radius:20px;font-size:0.72rem;font-weight:600">GAP</span>
            &nbsp; No adequate policy coverage found<br>
            <span style="background:#F1F5F9;color:#64748B;padding:2px 10px;
                border-radius:20px;font-size:0.72rem;font-weight:600">UNMATCHED</span>
            &nbsp; Definition, example, or not applicable
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("#### 🏛️ Supported Regulations & Policies")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""**Regulations (5)**
- AML Regulation (PMLA / VDA Guidelines)
- CERT-In Directions 2022
- DPDP Act 2023
- NBFC IT Framework (RBI)
- RBI Digital Payment Security Controls""")
    with col2:
        st.markdown("""**Company Policies (5)**
- Nirmal Bang — AML Policy
- Bajaj Finance — IT Policy
- Tata Motors — Data Protection
- Paytm — Cybersecurity Policy
- Razorpay — Digital Payment Security""")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYZE
# ══════════════════════════════════════════════════════════════════════════════
def page_analyze():
    st.markdown("""<div class="page-header">
        <h1>Run Analysis</h1>
        <p>Load a pre-computed report or upload new PDFs for live analysis</p>
    </div>""", unsafe_allow_html=True)

    IS_DEPLOYED = st.secrets.get("DEPLOYED", "false") == "true"

    if IS_DEPLOYED:
        tab1, = st.tabs(["📂 Load Pre-computed Report"])
    else:
        tab1, tab2 = st.tabs(["📂 Load Pre-computed Report", "📤 Upload New PDFs"])

    with tab1:
        st.markdown("#### Available Reports")
        output_dir   = Path("outputs")
        report_files = sorted(output_dir.glob("*.json"),
                               key=lambda p: p.stat().st_mtime,
                               reverse=True) if output_dir.exists() else []
        if not report_files:
            st.warning(
                "No report files found in `outputs/`.\n\n"
                "Run the analysis pipeline first:\n"
                "```bash\npython run_compliance_analysis.py --no-llm\n```"
            )
        else:
            options       = {p.name: str(p) for p in report_files[:15]}
            selected_name = st.selectbox("Select report", list(options.keys()),
                                          label_visibility="collapsed")
            selected_path = options[selected_name]
            p     = Path(selected_path)
            mtime = datetime.fromtimestamp(p.stat().st_mtime)
            st.caption(f"Modified: {mtime.strftime('%d %b %Y %H:%M')} · {p.stat().st_size//1024} KB")

            if st.button("Load This Report →"):
                with st.spinner("Loading…"):
                    normalized, raw, summary = load_and_normalize(selected_path)
                if normalized:
                    st.session_state.reg_items     = normalized
                    st.session_state.raw_report    = raw
                    st.session_state.report_data   = raw 
                    st.session_state.summary       = summary
                    st.session_state.analysis_done = True
                    st.session_state.page          = "results"
                    st.rerun()
                else:
                    st.error("Could not parse this report file.")

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("#### Generate a New Report via CLI")
        st.code(
            "# Heuristic mode — instant, no API:\n"
            "python run_compliance_analysis.py --no-llm\n\n"
            "# Full LLM mode (Groq):\n"
            "python run_compliance_analysis.py\n\n"
            "# Quick test on 20 items:\n"
            "python run_compliance_analysis.py --limit 20",
            language="bash"
        )

    if not IS_DEPLOYED:
        with tab2:
            st.info("Full pipeline takes 5–40 minutes depending on mode.")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📜 Regulation PDF**")
                reg_file = st.file_uploader("Regulation", type=["pdf"], key="reg_up",
                                             label_visibility="collapsed")
                if reg_file:
                    st.success(f"✅ {reg_file.name}")
                    st.session_state.regulation_file = reg_file
            with col2:
                st.markdown("**🏢 Company Policy PDF**")
                pol_file = st.file_uploader("Policy", type=["pdf"], key="pol_up",
                                             label_visibility="collapsed")
                if pol_file:
                    st.success(f"✅ {pol_file.name}")
                    st.session_state.policy_file = pol_file

            both = (st.session_state.regulation_file is not None and
                    st.session_state.policy_file is not None)
            if st.button("🔍 Analyze Compliance Gaps", disabled=not both):
                _run_pipeline_demo()


def _run_pipeline_demo():
    import requests
    import time

    reg_file = st.session_state.regulation_file
    pol_file = st.session_state.policy_file

    st.markdown("#### ⚙️ Pipeline Running")
    status_area = st.empty()

    status_area.markdown(
        "<div style='font-size:0.85rem;color:#0E7C7B;font-weight:500'>"
        "⟳ Submitting files to analysis pipeline…</div>",
        unsafe_allow_html=True
    )

    try:
        response = requests.post(
            "http://localhost:8000/analyze",
            files={
                "regulation_pdf": (reg_file.name, reg_file.getvalue(), "application/pdf"),
                "policy_pdf":     (pol_file.name, pol_file.getvalue(), "application/pdf"),
            },
            timeout=30
        )
        if response.status_code != 200:
            st.error(f"API error {response.status_code}: {response.text}")
            return

        job_id   = response.json()["job_id"]
        poll_url = f"http://localhost:8000/jobs/{job_id}"

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to FastAPI backend. "
            "Make sure it is running:\n\n"
            "```\nuvicorn api:app --reload --port 8000\n```"
        )
        return

    prog     = st.progress(0)
    pct      = 0

    step_labels = {
        "queued":                                        (5,  "Queued…"),
        "Ingesting regulation PDF into ChromaDB":        (15, "Ingesting regulation PDF…"),
        "Ingesting policy PDF into ChromaDB":            (30, "Ingesting policy PDF…"),
        "Running gap analysis":                          (50, "Running gap analysis…"),
        "Running compliance verification and LLM explanations": (75, "Running LLM verification…"),
        "Complete":                                      (100, "Complete ✅"),
    }

    max_wait      = 60 * 40
    poll_interval = 5
    elapsed       = 0

    while elapsed < max_wait:
        try:
            job = requests.get(poll_url, timeout=10).json()
        except Exception:
            time.sleep(poll_interval)
            elapsed += poll_interval
            continue

        step       = job.get("step", "")
        job_status = job.get("status", "")

        for key, (target_pct, label) in step_labels.items():
            if key in step or step == key:
                pct = target_pct
                break
        else:
            if job_status == "running":
                pct = min(pct + 1, 90)

        prog.progress(pct)
        status_area.markdown(
            f"<div style='font-size:0.85rem;color:#0E7C7B;font-weight:500'>"
            f"⟳ {step or 'Processing…'}</div>",
            unsafe_allow_html=True
        )

        if job_status == "done":
            prog.progress(100)
            result_filename = job.get("result")
            if not result_filename:
                st.error("Pipeline completed but no output file was returned.")
                return

            result_path = f"outputs/{result_filename}"
            normalized, raw, summary = load_and_normalize(result_path)
            if normalized:
                st.session_state.reg_items     = normalized
                st.session_state.raw_report    = raw
                st.session_state.report_data   = raw 
                st.session_state.summary       = summary
                st.session_state.analysis_done = True
                st.session_state.page          = "results"
                st.rerun()
            else:
                st.error(f"Could not parse output file: {result_path}")
            return

        if job_status == "error":
            st.error(f"Pipeline failed: {job.get('error', 'Unknown error')}")
            return

        time.sleep(poll_interval)
        elapsed += poll_interval

    st.error("Analysis timed out after 40 minutes. Check the API logs for details.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RESULTS SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def page_results():
    if not st.session_state.analysis_done or not st.session_state.summary:
        st.warning("No analysis loaded. Go to Run Analysis first.")
        if st.button("Go to Run Analysis"):
            st.session_state.page = "analyze"
            st.rerun()
        return

    s         = st.session_state.summary or {}
    all_items = st.session_state.reg_items or []

    total     = len(all_items)
    aligned   = sum(1 for i in all_items if i["classification"] == "aligned")
    partial   = sum(1 for i in all_items if i["classification"] == "partial")
    gap       = sum(1 for i in all_items if i["classification"] == "gap")
    unmatched = sum(1 for i in all_items if i["classification"] == "unmatched")
    coverage  = round(aligned / total * 100, 1) if total > 0 else 0

    st.markdown("""<div class="page-header">
        <h1>Compliance Summary</h1>
        <p>Overview of regulatory coverage across your policy documents</p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, color, val, label, sub in [
        (c1, "teal",  total,     "Total Regulations", "analyzed"),
        (c2, "green", aligned,   "Aligned",   f"{round(aligned/total*100,1) if total else 0}%"),
        (c3, "amber", partial,   "Partial",   f"{round(partial/total*100,1) if total else 0}%"),
        (c4, "red",   gap,       "Gaps",      f"{round(gap/total*100,1) if total else 0}%"),
        (c5, "teal",  unmatched, "N/A",       f"{round(unmatched/total*100,1) if total else 0}%"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-card {color}"><div class="label">{label}</div>'
                f'<div class="value">{val}</div><div class="sub">{sub}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    try:
        import plotly.graph_objects as go
        col_pie, col_bar = st.columns(2)

        with col_pie:
            st.markdown("##### Classification Distribution")
            fig = go.Figure(go.Pie(
                labels=["Aligned", "Partial", "Gap", "Unmatched"],
                values=[aligned, partial, gap, unmatched],
                hole=0.55,
                marker_colors=["#1E7E55", "#D4820A", "#C0392B", "#94A3B8"],
                textinfo="percent+label", textfont_size=12,
            ))
            fig.update_layout(
                showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=280,
                annotations=[dict(text=f"<b>{coverage}%</b><br>coverage",
                                  x=0.5, y=0.5, font_size=14, showarrow=False)]
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_bar:
            st.markdown("##### Coverage by Regulation")
            by_reg = s.get("by_regulation", {})
            if by_reg:
                labels, pcts, colors = [], [], []
                for reg, stats in sorted(by_reg.items()):
                    total_r = sum(stats.values())
                    comp    = stats.get("compliant", 0) + stats.get("substantially_compliant", 0)
                    pct     = round(comp / total_r * 100, 1) if total_r > 0 else 0
                    labels.append(reg[:30])
                    pcts.append(pct)
                    colors.append("#1E7E55" if pct >= 50 else "#D4820A" if pct >= 25 else "#C0392B")
                fig2 = go.Figure(go.Bar(
                    x=pcts, y=labels, orientation="h",
                    marker_color=colors,
                    text=[f"{p}%" for p in pcts], textposition="outside"
                ))
                fig2.update_layout(
                    height=280, margin=dict(t=10, b=10, l=10, r=10),
                    xaxis=dict(title="Coverage %", range=[0, 110]),
                    yaxis=dict(autorange="reversed"),
                    plot_bgcolor="white", paper_bgcolor="white",
                )
                st.plotly_chart(fig2, use_container_width=True)

    except ImportError:
        st.info("Install plotly for charts: `pip install plotly`")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("##### Risk Distribution")
    risk_counts = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
    for item in all_items:
        exp  = item.get("llm_explanation", {}) or {}
        risk = (exp.get("risk_level") or "unknown").lower()
        if risk not in risk_counts:
            risk = "unknown"
        risk_counts[risk] += 1

    rc1, rc2, rc3, rc4 = st.columns(4)
    color_map = {"high": "red", "medium": "amber", "low": "green", "unknown": "teal"}
    for col, (risk, count) in zip([rc1, rc2, rc3, rc4], risk_counts.items()):
        with col:
            st.markdown(
                f'<div class="metric-card {color_map[risk]}">'
                f'<div class="label">{risk.title()} Risk</div>'
                f'<div class="value">{count}</div>'
                f'<div class="sub">regulations</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    if st.button("🔍 Explore All Gaps →"):
        st.session_state.page = "explorer"
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: REGULATORY CHANGE MONITOR
# ══════════════════════════════════════════════════════════════════════════════

def page_change_monitor():
    st.markdown("""<div class="page-header">
        <h1>🔄 Regulatory Change Monitor</h1>
        <p>Upload an amended regulation PDF to detect new, modified, and removed obligations</p>
    </div>""", unsafe_allow_html=True)

    # ── Guard: ChromaDB must be initialised ──────────────────────────────────
    if st.session_state.chroma_collection is None:
        st.warning(
            "⚠️ ChromaDB collection is not initialised. "
            "Please load or run an analysis first so the regulation index is available.",
            icon="⚠️"
        )
        if "chroma_error" in st.session_state:
            st.code(st.session_state.chroma_error)
        if st.button("⚙️ Go to Run Analysis"):
            st.session_state.page = "analyze"
            st.rerun()
        return

    # ── Regulation selector ──────────────────────────────────────────────────
    REGULATION_OPTIONS = [
        "AML/PMLA",
        "CERT-In Directions 2022",
        "DPDP Act 2023",
        "NBFC IT Framework",
        "RBI Digital Payment Security Controls",
    ]

    col_sel, col_info = st.columns([2, 3])
    with col_sel:
        regulation = st.selectbox(
            "Select Regulation",
            options=REGULATION_OPTIONS,
            help="Choose the regulation whose updated version you are uploading.",
        )
    with col_info:
        st.markdown(
            '<div style="background:#E6F4F4;border-left:3px solid #0E7C7B;'
            'padding:0.75rem 1rem;border-radius:0 6px 6px 0;font-size:0.83rem;'
            'color:#1A2B3C;margin-top:1.6rem">'
            '📌 The uploaded PDF will be compared chunk-by-chunk against the '
            'version currently stored in ChromaDB.</div>',
            unsafe_allow_html=True
        )

    # ── File uploader ────────────────────────────────────────────────────────
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload Updated Regulation PDF",
        type="pdf",
        help="Upload the amended/revised version of the selected regulation.",
    )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    if not uploaded_file:
        st.info("📄 Upload an updated regulation PDF above to begin analysis.")
        return

    if not st.button("🔍 Analyse Changes", type="primary"):
        return

    # ── Run analysis ─────────────────────────────────────────────────────────
    with st.spinner("Comparing chunks against stored version…"):
        results = analyse_changes(
            new_pdf_bytes=uploaded_file.read(),
            regulation_name=regulation,
            chroma_collection=st.session_state.chroma_collection,
        )

    # ── Handle errors ─────────────────────────────────────────────────────────
    if results.get("error"):
        st.error(f"❌ Analysis failed: {results['error']}")
        return

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.markdown("#### 📊 Change Summary")
    c1, c2, c3, c4 = st.columns(4)
    for col, color, val, label, sub in [
        (c1, "red",   len(results["new"]),       "New",       "obligations added"),
        (c2, "amber", len(results["modified"]),  "Modified",  "wording changed"),
        (c3, "teal",  len(results["removed"]),   "Removed",   "obligations dropped"),
        (c4, "green", results["unchanged_count"],"Unchanged", "identical to stored"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-card {color}"><div class="label">{label}</div>'
                f'<div class="value">{val}</div><div class="sub">{sub}</div></div>',
                unsafe_allow_html=True
            )

    st.caption(
        f"New PDF: **{results['total_new_chunks']} chunks** | "
        f"Stored version: **{results['total_old_chunks']} chunks**"
    )

    # Nothing changed
    if (
        not results["new"]
        and not results["modified"]
        and not results["removed"]
    ):
        st.success(
            "✅ No significant changes detected. "
            "The uploaded PDF appears identical to the stored version."
        )
        return

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── New obligations ───────────────────────────────────────────────────────
    if results["new"]:
        st.markdown(f"#### 🆕 New Obligations &nbsp;<span style='font-size:0.85rem;color:#64748B'>({len(results['new'])})</span>", unsafe_allow_html=True)
        st.markdown(
            '<div class="text-block gap-block" style="margin-bottom:1rem">'
            'These obligations <strong>did not exist</strong> in the previously stored version. '
            'Review whether your policies need updating.</div>',
            unsafe_allow_html=True
        )
        for i, chunk in enumerate(results["new"], start=1):
            with st.expander(f"New obligation #{i}", expanded=(i <= 3)):
                st.markdown(
                    f'<div class="text-block gap-block">{chunk}</div>',
                    unsafe_allow_html=True
                )

    # ── Modified obligations ──────────────────────────────────────────────────
    if results["modified"]:
        st.markdown(f"#### ⚠️ Modified Obligations &nbsp;<span style='font-size:0.85rem;color:#64748B'>({len(results['modified'])})</span>", unsafe_allow_html=True)
        st.markdown(
            '<div class="text-block" style="border-color:var(--amber);margin-bottom:1rem">'
            'These obligations exist in <strong>both versions</strong> but the wording has changed. '
            'Sorted by degree of change — most changed first.</div>',
            unsafe_allow_html=True
        )
        sorted_modified = sorted(results["modified"], key=lambda x: x["similarity"])
        for i, item in enumerate(sorted_modified, start=1):
            change_label = "minor wording change" if item["similarity"] > 0.80 else "significant change"
            with st.expander(
                f"Modified #{i} — similarity: {item['similarity']:.2f} ({change_label})",
                expanded=(i <= 2)
            ):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📌 Stored Version**")
                    st.markdown(
                        f'<div class="text-block policy">{item["old"]}</div>',
                        unsafe_allow_html=True
                    )
                with col2:
                    st.markdown("**📝 New Version**")
                    st.markdown(
                        f'<div class="text-block regulation">{item["new"]}</div>',
                        unsafe_allow_html=True
                    )
                st.markdown(
                    score_bar(item["similarity"], color="#D4820A", label="Similarity Score"),
                    unsafe_allow_html=True
                )

    # ── Removed obligations ───────────────────────────────────────────────────
    if results["removed"]:
        st.markdown(f"#### ❌ Removed Obligations &nbsp;<span style='font-size:0.85rem;color:#64748B'>({len(results['removed'])})</span>", unsafe_allow_html=True)
        st.markdown(
            '<div class="text-block" style="border-color:var(--slate);margin-bottom:1rem">'
            'These obligations were in the stored version but have <strong>no counterpart</strong> '
            'in the uploaded PDF — they may have been repealed or restructured.</div>',
            unsafe_allow_html=True
        )
        for i, chunk in enumerate(results["removed"], start=1):
            with st.expander(f"Removed obligation #{i}", expanded=(i <= 3)):
                st.markdown(
                    f'<div class="text-block" style="border-color:var(--slate)">{chunk}</div>',
                    unsafe_allow_html=True
                )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.caption(
        "ℹ️ Thresholds — Unchanged: ≥ 0.90 | Modified: 0.60–0.89 | New: < 0.60"
    )


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════

render_sidebar()

page = st.session_state.page
if   page == "home":           page_home()
elif page == "analyze":        page_analyze()
elif page == "results":        page_results()
elif page == "explorer":       page_explorer_v2()
elif page == "export":         page_export_v2()
elif page == "change_monitor": page_change_monitor()   # ← NEW
else:                          page_home()