"""
REG-INSIGHT —  Detailed Results View & Evidence Display
Drop-in replacement for _render_detail() and page_explorer() in app.py

Changes from Week 7:
  - True side-by-side regulation vs policy comparison
  - All policy match candidates shown (not just top-1)
  - Keyword highlighting in text blocks
  - Section header breadcrumb
  - Chunk ID + page number metadata pills
  - Severity-based gap detail cards
  - Section-header filter in explorer
"""

import streamlit as st
import html as html_lib
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
#  ADDITIONAL CSS  (append this to your existing <style> block in app.py)
# ══════════════════════════════════════════════════════════════════════════════

WEEK8_CSS = """
<style>
/* ── Breadcrumb ─────────────────────────────────────────── */
.breadcrumb {
    display: flex; align-items: center; gap: 0.4rem;
    font-size: 0.72rem; color: #94A3B8;
    margin-bottom: 0.75rem; flex-wrap: wrap;
}
.breadcrumb .crumb {
    background: #F1F5F9; border-radius: 4px;
    padding: 0.15rem 0.5rem; color: #475569;
}
.breadcrumb .sep { color: #CBD5E1; }

/* ── Metadata pills ─────────────────────────────────────── */
.meta-pills { display: flex; gap: 0.4rem; flex-wrap: wrap; margin-bottom: 0.75rem; }
.meta-pill {
    background: #F8FAFC; border: 1px solid #E2E8F0;
    border-radius: 4px; padding: 0.15rem 0.55rem;
    font-size: 0.7rem; color: #64748B; font-family: 'DM Mono', monospace;
}

/* ── Side-by-side comparison ────────────────────────────── */
.compare-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;
    margin-bottom: 1rem;
}
.compare-col { display: flex; flex-direction: column; }
.compare-label {
    font-size: 0.68rem; text-transform: uppercase;
    letter-spacing: 0.1em; font-weight: 600;
    margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.4rem;
}
.compare-label.reg-label { color: #0E7C7B; }
.compare-label.pol-label { color: #D4820A; }
.compare-box {
    flex: 1; padding: 0.85rem 1rem; border-radius: 8px;
    font-size: 0.82rem; line-height: 1.65; color: #1A2B3C;
    border-left: 3px solid; min-height: 120px;
    background: #F8FAFC;
}
.compare-box.reg { border-color: #0E7C7B; }
.compare-box.pol { border-color: #D4820A; }
.compare-box mark {
    background: #FEF9C3; color: #92400E;
    border-radius: 2px; padding: 0 2px;
}

/* ── Match candidate cards ──────────────────────────────── */
.candidate-card {
    border: 1px solid #E2E8F0; border-radius: 8px;
    padding: 0.75rem 1rem; margin-bottom: 0.5rem;
    background: white; position: relative;
}
.candidate-card.top-match { border-color: #0E7C7B; border-width: 2px; }
.candidate-rank {
    position: absolute; top: 0.6rem; right: 0.75rem;
    font-size: 0.68rem; color: #94A3B8; font-family: 'DM Mono', monospace;
}
.candidate-text { font-size: 0.81rem; color: #1A2B3C; line-height: 1.6; margin-bottom: 0.5rem; }
.candidate-meta { font-size: 0.72rem; color: #64748B; }

/* ── Severity card ──────────────────────────────────────── */
.severity-card {
    border-radius: 10px; padding: 1rem 1.25rem;
    margin-bottom: 0.75rem; border: 1px solid;
}
.severity-card.high   { background: #FEF2F2; border-color: #FECACA; }
.severity-card.medium { background: #FFFBEB; border-color: #FDE68A; }
.severity-card.low    { background: #F0FDF4; border-color: #BBF7D0; }
.severity-card .sev-title {
    font-weight: 600; font-size: 0.85rem; margin-bottom: 0.3rem;
}
.severity-card.high   .sev-title { color: #991B1B; }
.severity-card.medium .sev-title { color: #92400E; }
.severity-card.low    .sev-title { color: #166534; }
.severity-card .sev-body { font-size: 0.82rem; line-height: 1.6; color: #374151; }

/* ── Score comparison table ─────────────────────────────── */
.score-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
.score-table th {
    text-align: left; font-size: 0.68rem; text-transform: uppercase;
    letter-spacing: 0.08em; color: #94A3B8; padding: 0.3rem 0.5rem;
    border-bottom: 1px solid #E2E8F0;
}
.score-table td { padding: 0.4rem 0.5rem; color: #1A2B3C; }
.score-table tr:hover td { background: #F8FAFC; }
</style>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _highlight(text: str, query: str) -> str:
    """Wrap query terms in <mark> tags (case-insensitive)."""
    if not query or not text:
        return html_lib.escape(text or "")
    escaped = html_lib.escape(text)
    import re
    for term in query.split():
        if len(term) < 2:
            continue
        pattern = re.compile(re.escape(html_lib.escape(term)), re.IGNORECASE)
        escaped = pattern.sub(lambda m: f"<mark>{m.group()}</mark>", escaped)
    return escaped


def _score_color(score: float) -> str:
    if score >= 0.70:
        return "#1E7E55"
    if score >= 0.40:
        return "#D4820A"
    return "#C0392B"


def _breadcrumb_html(item: dict) -> str:
    meta   = item.get("regulation_metadata") or {}
    source = item.get("regulation_source") or item.get("source", "")
    parts  = []

    src_name = Path(source).stem if source else ""
    if src_name:
        parts.append(f'<span class="crumb">{html_lib.escape(src_name)}</span>')

    domain = meta.get("domain") or item.get("category", "")
    if domain and domain not in ("N/A", ""):
        parts.append('<span class="sep">›</span>')
        parts.append(f'<span class="crumb">{html_lib.escape(str(domain))}</span>')

    section = meta.get("section_header", "")
    if section:
        parts.append('<span class="sep">›</span>')
        parts.append(f'<span class="crumb">{html_lib.escape(str(section)[:60])}</span>')

    if not parts:
        return ""
    return f'<div class="breadcrumb">{"".join(parts)}</div>'


def _meta_pills_html(item: dict) -> str:
    meta    = item.get("regulation_metadata") or {}
    pills   = []
    chunk_id = item.get("regulation_chunk_id", "")
    page     = meta.get("page_number", "")

    if chunk_id:
        pills.append(f'<span class="meta-pill">chunk: {html_lib.escape(str(chunk_id)[-12:])}</span>')
    if page:
        pills.append(f'<span class="meta-pill">page: {page}</span>')
    score = item.get("final_score") or item.get("best_score") or 0
    pills.append(f'<span class="meta-pill">score: {float(score):.3f}</span>')

    return f'<div class="meta-pills">{"".join(pills)}</div>' if pills else ""


def _severity_card_html(exp: dict, classification: str) -> str:
    risk    = (exp.get("risk_level") or "low").lower()
    gap_desc = html_lib.escape(exp.get("gap_description") or "")
    summary  = html_lib.escape(exp.get("summary") or "")

    icon_map = {"high": "🔴", "medium": "🟡", "low": "🟢"}
    icon     = icon_map.get(risk, "⚪")
    title    = f"{icon} {risk.title()} Risk — {classification.title()}"
    body     = gap_desc or summary or "No description available."

    return f"""
    <div class="severity-card {risk}">
        <div class="sev-title">{title}</div>
        <div class="sev-body">{body}</div>
    </div>"""


def score_bar_inline(score: float, color: str, label: str) -> str:
    pct = int((score or 0) * 100)
    return f"""
    <div style="margin:0.25rem 0 0.6rem">
        <div style="display:flex;justify-content:space-between;
                    font-size:0.72rem;color:#64748B;margin-bottom:3px">
            <span>{label}</span>
            <span style="font-family:'DM Mono',monospace;color:{color}">{score:.3f}</span>
        </div>
        <div style="height:5px;background:#E2E8F0;border-radius:3px;overflow:hidden">
            <div style="width:{pct}%;height:100%;background:{color};border-radius:3px"></div>
        </div>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN DETAIL RENDERER  (replaces _render_detail in app.py)
# ══════════════════════════════════════════════════════════════════════════════

import re

def _strip_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&nbsp;',  ' ', text)
    text = re.sub(r'&amp;',   '&', text)
    text = re.sub(r'\s+',     ' ', text).strip()
    return text

def render_detail_v2(item: dict, search_query: str = ""):
    """
    Full Week-8 detail panel.

    Parameters
    ----------
    item         : normalized regulation analysis item
    search_query : active search string from the explorer filter bar
                   (used to highlight keywords in text blocks)
    """
    cls   = item.get("classification", "unmatched")
    score = float(item.get("final_score") or item.get("best_score") or 0)
    bi    = float(item.get("bi_encoder_score") or 0)
    ce    = float(item.get("cross_encoder_score") or 0)
    exp   = item.get("llm_explanation") or {}

    reg_text   = item.get("regulation_text", "No text available")
    reg_source = item.get("regulation_source") or item.get("source", "")
    matches    = item.get("policy_matches") or []

    top_match  = matches[0] if matches else {}
    pol_text = _strip_html(
    top_match.get("policy_text") or
    top_match.get("policy_chunk_preview") or
    item.get("policy_text") or
    ""
) or "No matching policy text found."
    pol_source = (
        top_match.get("policy_source") or
        item.get("policy_document") or
        item.get("matched_policy") or ""
    )
    pol_stem   = Path(pol_source).stem if pol_source else "—"

    # ── Breadcrumb + pills ────────────────────────────────────────────────
    st.markdown(_breadcrumb_html(item),    unsafe_allow_html=True)
    st.markdown(_meta_pills_html(item),    unsafe_allow_html=True)

    # ── Severity card ─────────────────────────────────────────────────────
    st.markdown(_severity_card_html(exp, cls), unsafe_allow_html=True)

    # ── Side-by-side comparison ───────────────────────────────────────────
    reg_highlighted = _highlight(reg_text,  search_query)
    pol_highlighted = _highlight(pol_text[:800], search_query)
    pol_ellipsis    = "…" if len(pol_text) > 800 else ""
    pol_src_label = f"<span style='font-size:0.7rem;color:#94A3B8'> · {html_lib.escape(pol_stem or '')}</span>"

    st.markdown(f"""
    <div class="compare-grid">
        <div class="compare-col">
            <div class="compare-label reg-label">
                <span>⚖️</span> Regulation Requirement
            </div>
            <div class="compare-box reg">{reg_highlighted}</div>
        </div>
        <div class="compare-col">
            <div class="compare-label pol-label">
                <span>📋</span> Best Matching Policy {pol_src_label}
            </div>
            <div class="compare-box pol">{pol_highlighted}{pol_ellipsis}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Score indicators ──────────────────────────────────────────────────
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.markdown(
            score_bar_inline(score, _score_color(score), "Final Score"),
            unsafe_allow_html=True
        )
    with col_s2:
        st.markdown(
            score_bar_inline(bi, "#64748B", "Bi-encoder"),
            unsafe_allow_html=True
        )
    with col_s3:
        st.markdown(
            score_bar_inline(ce, "#64748B", "Cross-encoder"),
            unsafe_allow_html=True
        )

    # ── All policy match candidates ───────────────────────────────────────
    if len(matches) > 1:
        with st.expander(f"📎 All {len(matches)} Policy Candidates", expanded=False):
            for rank, match in enumerate(matches, 1):
                m_text = _strip_html(match.get("policy_text") or match.get("policy_chunk_preview") or "")
                m_source = match.get("policy_source") or ""
                m_score  = float(match.get("score") or match.get("final_score") or 0)
                m_stem   = Path(m_source).stem if m_source else "—"
                is_top   = rank == 1
                preview  = html_lib.escape((m_text or "")[:300]) + ("…" if len(m_text or "") > 300 else "")

                st.markdown(f"""
                <div class="candidate-card {'top-match' if is_top else ''}">
                    <div class="candidate-rank">#{rank} &nbsp; score: {m_score:.3f}</div>
                    <div class="candidate-text">{preview}</div>
                    <div class="candidate-meta">
                        📄 {html_lib.escape(m_stem or '')}
                        {'&nbsp;·&nbsp;<b style="color:#0E7C7B">Top Match</b>' if is_top else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col_bar, _ = st.columns([3, 1])
                with col_bar:
                    st.markdown(
                        score_bar_inline(m_score, _score_color(m_score), "Score"),
                        unsafe_allow_html=True
                    )

    # ── LLM Explanation ───────────────────────────────────────────────────
    summary = exp.get("summary", "")
    action  = exp.get("recommended_action", "") or item.get("recommended_action", "")

    if summary:
        st.markdown(f"""
        <div class="explanation-card" style="margin-top:0.75rem">
            <div class="exp-title">🤖 AI Explanation</div>
            <div class="exp-body">{html_lib.escape(summary or '')}</div>
        </div>
        """, unsafe_allow_html=True)

    what_covers = item.get("what_policy_covers", "")
    if what_covers:
        st.markdown(f"""
        <div style="margin-top:0.75rem;padding:0.75rem 1rem;
                    background:#F0F7FF;border:1px solid #B8D9F0;
                    border-radius:8px;font-size:0.83rem;color:#1A3A5C">
            <b>📋 What Policy Covers:</b> {html_lib.escape(what_covers or '')}
        </div>
        """, unsafe_allow_html=True)

    if action:
        st.markdown(f"""
        <div style="margin-top:0.75rem;padding:0.75rem 1rem;
                    background:#F0FDF4;border:1px solid #BBF7D0;
                    border-radius:8px;font-size:0.83rem;color:#1E7E55">
            <b>✅ Recommended Action:</b> {html_lib.escape(action or '')}
        </div>
        """, unsafe_allow_html=True)

    if not summary and not action:
        st.markdown("""
        <div style="padding:0.75rem 1rem;background:#F8FAFC;
                    border:1px solid #E2E8F0;border-radius:8px;
                    font-size:0.83rem;color:#94A3B8">
            No AI explanation available for this item.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  UPGRADED GAP EXPLORER PAGE  (replaces page_explorer in app.py)
# ══════════════════════════════════════════════════════════════════════════════

def page_explorer_v2():
    """
    Week-8 Gap Explorer with:
      - section-header filter
      - severity (risk level) filter
      - keyword search with highlighting
      - improved list cards
      - side-by-side detail panel via render_detail_v2
    """
    if not st.session_state.analysis_done or not st.session_state.reg_items:
        st.warning("No analysis loaded.")
        return

    # Inject Week-8 CSS
    st.markdown(WEEK8_CSS, unsafe_allow_html=True)

    all_items = st.session_state.reg_items or []

    st.markdown("""<div class="page-header">
        <h1>Gap Explorer</h1>
        <p>Browse, filter and inspect every regulation requirement</p>
    </div>""", unsafe_allow_html=True)

    # ── Collect unique section headers for filter ─────────────────────────
    sections = set()
    for item in all_items:
        meta = item.get("regulation_metadata") or {}
        sec  = meta.get("section_header", "")
        if sec and sec not in ("", "N/A"):
            sections.add(str(sec)[:60])
    section_options = ["All"] + sorted(sections)

    # ── Filter row ────────────────────────────────────────────────────────
    fc1, fc2, fc3, fc4 = st.columns([2.5, 1.2, 1.2, 1.5])
    with fc1:
        search = st.text_input(
            "🔍 Search",
            placeholder="e.g. KYC, data breach, incident report…",
            key="explorer_search"
        )
    with fc2:
        cls_filter = st.selectbox(
            "Classification",
            ["All", "Gap", "Partial", "Aligned", "Unmatched"],
            key="explorer_cls"
        )
    with fc3:
        risk_filter = st.selectbox(
            "Risk Level",
            ["All", "High", "Medium", "Low"],
            key="explorer_risk"
        )
    with fc4:
        section_filter = st.selectbox(
            "Section",
            section_options,
            key="explorer_section"
        )

    # ── Apply filters ─────────────────────────────────────────────────────
    filtered = all_items

    if cls_filter != "All":
        filtered = [i for i in filtered
                    if i.get("classification", "").lower() == cls_filter.lower()]

    if risk_filter != "All":
        filtered = [i for i in filtered
                    if (i.get("llm_explanation") or {}).get("risk_level", "").lower()
                    == risk_filter.lower()]

    if section_filter != "All":
        filtered = [i for i in filtered
                    if str((i.get("regulation_metadata") or {}).get("section_header", ""))[:60]
                    == section_filter]

    if search:
        q        = search.lower()
        filtered = [i for i in filtered
                    if q in i.get("regulation_text", "").lower()
                    or q in (i.get("regulation_source") or i.get("source", "")).lower()
                    or q in str((i.get("regulation_metadata") or {}).get("section_header", "")).lower()]

    st.markdown(
        f"<div style='font-size:0.82rem;color:#64748B;margin-bottom:0.75rem'>"
        f"Showing <b>{len(filtered)}</b> of <b>{len(all_items)}</b> regulations</div>",
        unsafe_allow_html=True
    )

    # ── Pagination ────────────────────────────────────────────────────────
    PAGE_SIZE = 30
    total_pages = max(1, (len(filtered) + PAGE_SIZE - 1) // PAGE_SIZE)

    if "explorer_page" not in st.session_state:
        st.session_state.explorer_page = 0
    # Reset page when filters change
    filter_key = f"{cls_filter}|{risk_filter}|{section_filter}|{search}"
    if st.session_state.get("_last_filter_key") != filter_key:
        st.session_state.explorer_page = 0
        st.session_state["_last_filter_key"] = filter_key

    page_start = st.session_state.explorer_page * PAGE_SIZE
    page_items = filtered[page_start: page_start + PAGE_SIZE]

    # ── Split view ────────────────────────────────────────────────────────
    list_col, detail_col = st.columns([2, 3])

    with list_col:
        st.markdown("##### Regulations")

        if not filtered:
            st.info("No items match your filters.")
        else:
            for idx, item in enumerate(page_items):
                global_idx = page_start + idx
                cls        = item.get("classification", "unmatched")
                text       = item.get("regulation_text", "")[:100]
                source     = item.get("regulation_source") or item.get("source", "")
                score      = float(item.get("final_score") or item.get("best_score") or 0)
                exp        = item.get("llm_explanation") or {}
                risk       = (exp.get("risk_level") or "").lower()
                meta       = item.get("regulation_metadata") or {}
                section    = str(meta.get("section_header", ""))[:40]

                is_selected = st.session_state.selected_item_idx == global_idx
                border      = "border:2px solid #0E7C7B;" if is_selected else ""

                # Inline badge colours
                cls_colors = {
                    "aligned":   "background:#E8F5EE;color:#1E7E55",
                    "partial":   "background:#FDF3E0;color:#D4820A",
                    "gap":       "background:#FDECEA;color:#C0392B",
                    "unmatched": "background:#F1F5F9;color:#64748B",
                }
                risk_colors = {
                    "high":   "background:#FDECEA;color:#C0392B",
                    "medium": "background:#FDF3E0;color:#D4820A",
                    "low":    "background:#E8F5EE;color:#1E7E55",
                }
                badge_style = cls_colors.get(cls, "background:#F1F5F9;color:#64748B")
                risk_style  = risk_colors.get(risk, "")
                score_color = _score_color(score)

                badge_span = (
                    f'<span style="{badge_style};display:inline-block;'
                    f'padding:0.15rem 0.55rem;border-radius:20px;'
                    f'font-size:0.68rem;font-weight:600;text-transform:uppercase">'
                    f'{cls.title()}</span>'
                )
                risk_span = (
                    f'<span style="{risk_style};display:inline-block;'
                    f'padding:0.15rem 0.55rem;border-radius:20px;'
                    f'font-size:0.68rem;font-weight:600;text-transform:uppercase">'
                    f'{risk.title()} Risk</span>'
                    if risk and risk_style else ""
                )
                section_span = (
                    f'<span style="font-size:0.7rem;color:#94A3B8">'
                    f'{html_lib.escape(section)}</span>'
                    if section else ""
                )

                # ── FIX: score rendered directly in span style, no nested f-string ──
                score_span = (
                    f'<span style="color:{score_color};font-size:0.7rem;'
                    f'font-family:\'DM Mono\',monospace;margin-left:auto">'
                    f'{score:.2f}</span>'
                )

                st.markdown(f"""
                <div class="result-row" style="{border}">
                    <div class="reg-text">{html_lib.escape(text or '')}…</div>
                    <div class="meta" style="margin-top:0.3rem;display:flex;
                                            flex-wrap:wrap;gap:0.3rem;align-items:center">
                        {badge_span}{risk_span}{section_span}{score_span}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if st.button("View →", key=f"sel_{global_idx}", use_container_width=True):
                    st.session_state.selected_item_idx = global_idx
                    st.rerun()

            # Pagination controls
            if total_pages > 1:
                st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                p_col1, p_col2, p_col3 = st.columns([1, 2, 1])
                with p_col1:
                    if st.button("← Prev", disabled=st.session_state.explorer_page == 0):
                        st.session_state.explorer_page -= 1
                        st.rerun()
                with p_col2:
                    st.markdown(
                        f"<div style='text-align:center;font-size:0.8rem;color:#64748B;"
                        f"padding-top:0.5rem'>Page {st.session_state.explorer_page + 1} "
                        f"of {total_pages}</div>",
                        unsafe_allow_html=True
                    )
                with p_col3:
                    if st.button("Next →", disabled=st.session_state.explorer_page >= total_pages - 1):
                        st.session_state.explorer_page += 1
                        st.rerun()

    # ── Detail panel ──────────────────────────────────────────────────────
    with detail_col:
        st.markdown("##### Detail View")
        idx = st.session_state.selected_item_idx
        if idx is None or idx >= len(filtered):
            st.markdown("""
            <div style="text-align:center;padding:3rem;color:#94A3B8">
                <div style="font-size:2rem">←</div>
                <div style="font-size:0.88rem">Select a regulation to see details</div>
            </div>""", unsafe_allow_html=True)
        else:
            render_detail_v2(filtered[idx], search_query=search)