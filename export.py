"""
REG-INSIGHT — Week 9: Export & Report Generation
Drop-in replacement / addition for page_export() in app.py

New features over Week 7:
  - Markdown report with executive summary + regulation tables
  - Analysis history (save/load snapshots to outputs/snapshots/)
  - Document hash + audit trail embedded in every export
  - Print-friendly summary view
"""

import streamlit as st
import json
import csv
import io
import hashlib
import os
from pathlib import Path
from datetime import datetime


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _get_summary_stats(report: dict) -> tuple:
    s         = report.get("summary", {})
    cls       = s.get("classifications", {})
    total     = s.get("total_regulations", s.get("total_processed", 0))
    aligned   = cls.get("aligned", 0)
    partial   = cls.get("partial", 0)
    gap       = cls.get("gap", 0)
    unmatched = cls.get("unmatched", 0)
    coverage  = s.get("coverage_percentage", 0)
    return total, aligned, partial, gap, unmatched, coverage


def _report_hash(report: dict) -> str:
    """Stable MD5 of the report JSON (used as audit fingerprint)."""
    raw = json.dumps(report, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(raw.encode()).hexdigest()


def _snapshots_dir() -> Path:
    d = Path("outputs/snapshots")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_snapshot(report: dict, label: str = "") -> Path:
    """Write report to outputs/snapshots/ with timestamp + hash."""
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    h    = _report_hash(report)[:8]
    slug = label.replace(" ", "_")[:30] if label else "snapshot"
    filename = f"{slug}_{ts}_{h}.json"
    path     = _snapshots_dir() / filename

    export = {
        **report,
        "_audit": {
            "generated_at":  datetime.now().isoformat(),
            "report_hash":   _report_hash(report),
            "snapshot_label": label or filename,
            "tool":          "REG-INSIGHT",
        }
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)
    return path


def _list_snapshots() -> list[Path]:
    d = _snapshots_dir()
    return sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def _load_snapshot(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
#  MARKDOWN REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_markdown_report(report: dict) -> str:
    total, aligned, partial, gap, unmatched, coverage = _get_summary_stats(report)
    items    = report.get("regulation_analysis", [])
    h        = _report_hash(report)
    ts       = datetime.now().strftime("%d %B %Y, %H:%M")
    gap_only = [i for i in items if i.get("classification") in ("gap", "unmatched")]

    # ── Risk breakdown ────────────────────────────────────────────────────
    risk_counts = {"high": 0, "medium": 0, "low": 0}
    for item in items:
        exp  = item.get("llm_explanation") or {}
        risk = (exp.get("risk_level") or "low").lower()
        if risk in risk_counts:
            risk_counts[risk] += 1

    # ── Per-regulation breakdown ──────────────────────────────────────────
    reg_stats: dict[str, dict] = {}
    for item in items:
        src = Path(
            item.get("regulation_source") or item.get("source") or "Unknown"
        ).stem
        cls = item.get("classification", "unmatched")
        if src not in reg_stats:
            reg_stats[src] = {"aligned": 0, "partial": 0, "gap": 0, "unmatched": 0}
        if cls in reg_stats[src]:
            reg_stats[src][cls] += 1

    # ── Build Markdown ────────────────────────────────────────────────────
    lines = []
    lines.append("# REG-INSIGHT — Compliance Gap Analysis Report")
    lines.append(f"\n**Generated:** {ts}  \n**Report Hash:** `{h}`  \n**Tool:** REG-INSIGHT v1.0\n")
    lines.append("---\n")

    # Executive summary
    lines.append("## Executive Summary\n")
    lines.append(
        f"Analysis of **{total}** regulatory obligation chunks across "
        f"**{len(reg_stats)}** regulation documents identified a compliance "
        f"coverage of **{coverage}%**. "
        f"**{gap}** obligations represent confirmed gaps and **{unmatched}** "
        f"had no matching policy content at all, requiring immediate attention."
    )
    lines.append("\n")

    # KPI table
    lines.append("| Metric | Count | % of Total |")
    lines.append("|--------|------:|-----------:|")
    for label, val in [
        ("✅ Aligned",    aligned),
        ("🟡 Partial",    partial),
        ("🔴 Gap",        gap),
        ("⚪ Unmatched",  unmatched),
        ("**Total**",     total),
    ]:
        pct = f"{round(val/total*100,1)}%" if total else "—"
        lines.append(f"| {label} | {val} | {pct} |")

    lines.append("\n")

    # Risk distribution
    lines.append("### Risk Distribution\n")
    lines.append("| Risk Level | Count |")
    lines.append("|------------|------:|")
    for risk, count in risk_counts.items():
        lines.append(f"| {risk.title()} | {count} |")
    lines.append("\n")

    # Per-regulation breakdown
    lines.append("---\n")
    lines.append("## Coverage by Regulation\n")
    lines.append("| Regulation | Aligned | Partial | Gap | Unmatched | Total | Coverage % |")
    lines.append("|------------|--------:|--------:|----:|----------:|------:|-----------:|")
    for reg, counts in sorted(reg_stats.items()):
        t   = sum(counts.values())
        cov = round((counts["aligned"] + counts["partial"]) / t * 100, 1) if t else 0
        lines.append(
            f"| {reg} "
            f"| {counts['aligned']} "
            f"| {counts['partial']} "
            f"| {counts['gap']} "
            f"| {counts['unmatched']} "
            f"| {t} "
            f"| {cov}% |"
        )
    lines.append("\n")

    # Prioritised gap list
    lines.append("---\n")
    lines.append("## Prioritised Gap List\n")
    lines.append(
        f"The following **{len(gap_only)}** obligations require policy remediation, "
        "ordered by risk level (High → Medium → Low).\n"
    )

    def _risk_order(item):
        r = (item.get("llm_explanation") or {}).get("risk_level", "low").lower()
        return {"high": 0, "medium": 1, "low": 2}.get(r, 3)

    sorted_gaps = sorted(gap_only, key=_risk_order)

    for i, item in enumerate(sorted_gaps, 1):
        exp    = item.get("llm_explanation") or {}
        risk   = (exp.get("risk_level") or "unknown").upper()
        cls    = item.get("classification", "unmatched").title()
        src    = Path(item.get("regulation_source") or item.get("source") or "").stem
        score  = float(item.get("final_score") or item.get("best_score") or 0)
        text   = item.get("regulation_text", "")[:200].replace("\n", " ")
        summary = exp.get("summary", "")
        action  = exp.get("recommended_action") or item.get("recommended_action") or ""

        lines.append(f"### {i}. [{risk}] {cls} — {src}\n")
        lines.append(f"**Similarity Score:** {score:.3f}  ")
        lines.append(f"\n**Regulation Text:**  \n> {text}{'…' if len(item.get('regulation_text',''))>200 else ''}\n")

        if summary:
            lines.append(f"**AI Analysis:**  \n{summary}\n")
        if action:
            lines.append(f"**Recommended Action:**  \n{action}\n")
        lines.append("---\n")

    # Appendix: methodology
    lines.append("## Appendix: Methodology\n")
    lines.append("""
This report was generated by **REG-INSIGHT**, an automated regulatory compliance
gap detection system using a multi-stage NLP pipeline:

1. **PDF Ingestion** — PyMuPDF text extraction with RecursiveCharacterTextSplitter (chunk_size=512)
2. **Semantic Embedding** — all-MiniLM-L6-v2 bi-encoder (384-dimensional vectors)
3. **Hybrid Retrieval** — BM25 (30%) + semantic search (70%) via ChromaDB
4. **Cross-encoder Reranking** — ms-marco-MiniLM-L6-v2 for precise pairwise scoring
5. **Gap Classification** — Aligned ≥ 0.70 · Partial 0.40–0.69 · Gap < 0.40
6. **LLM Explanation** — Groq API (llama3-8b-8192) with few-shot prompting

_This report is auto-generated. All classifications should be reviewed by a
qualified compliance professional before regulatory submission._
""")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  AUDIT-ENRICHED JSON EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def _enriched_json(report: dict) -> str:
    export = {
        **report,
        "_audit": {
            "generated_at": datetime.now().isoformat(),
            "report_hash":  _report_hash(report),
            "tool":         "REG-INSIGHT",
            "version":      "1.0",
        }
    }
    return json.dumps(export, indent=2, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
#  FULL EXPORT PAGE  (replaces page_export in app.py)
# ══════════════════════════════════════════════════════════════════════════════

def page_export_v2():
    if not st.session_state.analysis_done or not st.session_state.report_data:
        st.warning("No analysis loaded. Go to **Run Analysis** first.")
        return

    report = st.session_state.report_data
    items  = report.get("regulation_analysis", [])
    total, aligned, partial, gap, unmatched, coverage = _get_summary_stats(report)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M")
    h      = _report_hash(report)[:8]

    st.markdown("""<div class="page-header">
        <h1>Export Results</h1>
        <p>Download your compliance analysis in multiple formats</p>
    </div>""", unsafe_allow_html=True)

    st.caption(f"Report fingerprint: `{h}`  ·  {total} regulations  ·  {coverage}% coverage")

    # ══ TAB LAYOUT ══════════════════════════════════════════════════════
    tab_dl, tab_history, tab_print = st.tabs([
        "⬇️ Download Reports",
        "🕐 Analysis History",
        "🖨️ Print View",
    ])

    # ── TAB 1: Downloads ──────────────────────────────────────────────────
    with tab_dl:
        col1, col2, col3 = st.columns(3)

        # JSON
        with col1:
            st.markdown("#### 📄 Full JSON Report")
            st.markdown(
                "Complete analysis with all scores, policy matches, "
                "LLM explanations, and audit trail."
            )
            st.download_button(
                label="⬇️ Download JSON",
                data=_enriched_json(report),
                file_name=f"reg_insight_{ts_str}_{h}.json",
                mime="application/json",
                use_container_width=True
            )

        # Markdown
        with col2:
            st.markdown("#### 📝 Markdown Report")
            st.markdown(
                "Formatted report with executive summary, per-regulation "
                "tables, and prioritised gap list. Suitable for stakeholder sharing."
            )
            md_content = generate_markdown_report(report)
            st.download_button(
                label="⬇️ Download Markdown",
                data=md_content,
                file_name=f"reg_insight_{ts_str}_{h}.md",
                mime="text/markdown",
                use_container_width=True
            )

        # CSV
        with col3:
            st.markdown("#### 📊 Full CSV")
            st.markdown(
                "One row per regulation — suitable for Excel, Jira, "
                "or any compliance tracking system."
            )
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow([
                "Priority", "Regulation Text", "Regulation Source",
                "Section Header", "Page Number",
                "Classification", "Final Score",
                "Bi-encoder Score", "Cross-encoder Score",
                "Risk Level", "AI Summary", "Recommended Action",
                "Policy Source", "Report Hash",
            ])
            gap_counter = 0
            for item in items:
                exp    = item.get("llm_explanation") or {}
                meta   = item.get("regulation_metadata") or {}
                matches = item.get("policy_matches") or []
                top     = matches[0] if matches else {}
                pol_src = (
                    top.get("policy_source") or
                    item.get("policy_document") or
                    item.get("matched_policy") or ""
                )
                reg_src = item.get("regulation_source") or item.get("source", "")
                cls     = item.get("classification", "")
                if cls in ("gap", "unmatched"):
                    gap_counter += 1
                    priority = gap_counter
                else:
                    priority = ""

                writer.writerow([
                    priority,
                    item.get("regulation_text", "")[:250],
                    Path(reg_src).name if reg_src else "",
                    meta.get("section_header", ""),
                    meta.get("page_number", ""),
                    cls,
                    f"{float(item.get('final_score') or item.get('best_score') or 0):.3f}",
                    f"{float(item.get('bi_encoder_score') or 0):.3f}",
                    f"{float(item.get('cross_encoder_score') or 0):.3f}",
                    exp.get("risk_level", ""),
                    exp.get("summary", ""),
                    exp.get("recommended_action", "") or item.get("recommended_action", ""),
                    Path(pol_src).name if pol_src else "",
                    h,
                ])
            st.download_button(
                label="⬇️ Download CSV",
                data=output.getvalue(),
                file_name=f"reg_insight_full_{ts_str}.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.markdown("<hr style='margin:1.5rem 0;border-color:#E2E8F0'>", unsafe_allow_html=True)

        # Gaps-only CSV
        st.markdown("#### 🔴 Gaps-Only Report")
        st.markdown("Only gap and unmatched items — focused remediation list for compliance teams.")

        gap_items = [i for i in items if i.get("classification") in ("gap", "unmatched")]
        out2      = io.StringIO()
        w2        = csv.writer(out2)
        w2.writerow([
            "Priority", "Classification", "Risk Level",
            "Regulation Text", "Regulation Source", "Section Header",
            "Best Policy Match (Preview)", "Policy Source",
            "Final Score", "Recommended Action", "Report Hash",
        ])
        for priority, item in enumerate(gap_items, 1):
            exp     = item.get("llm_explanation") or {}
            meta    = item.get("regulation_metadata") or {}
            matches = item.get("policy_matches") or []
            top     = matches[0] if matches else {}
            pol_src = (
                top.get("policy_source") or
                item.get("policy_document") or
                item.get("matched_policy") or ""
            )
            reg_src = item.get("regulation_source") or item.get("source", "")
            w2.writerow([
                priority,
                item.get("classification", "").upper(),
                (exp.get("risk_level") or "unknown").upper(),
                item.get("regulation_text", "")[:250],
                Path(reg_src).name if reg_src else "",
                meta.get("section_header", ""),
                (top.get("policy_text") or item.get("policy_text") or "")[:150],
                Path(pol_src).name if pol_src else "",
                f"{float(item.get('final_score') or item.get('best_score') or 0):.3f}",
                exp.get("recommended_action", "") or item.get("recommended_action", ""),
                h,
            ])

        c_gap1, c_gap2, _ = st.columns([1, 1, 2])
        with c_gap1:
            st.metric("Gap + Unmatched Items", len(gap_items))
        with c_gap2:
            st.download_button(
                label="⬇️ Download Gaps CSV",
                data=out2.getvalue(),
                file_name=f"reg_insight_gaps_{ts_str}.csv",
                mime="text/csv",
                use_container_width=True
            )

    # ── TAB 2: Analysis History ───────────────────────────────────────────
    with tab_history:
        st.markdown("#### 💾 Save Current Analysis")

        save_col1, save_col2 = st.columns([3, 1])
        with save_col1:
            snap_label = st.text_input(
                "Snapshot label (optional)",
                placeholder="e.g. baseline_before_policy_update",
                key="snap_label"
            )
        with save_col2:
            st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
            if st.button("💾 Save Snapshot", use_container_width=True):
                try:
                    saved_path = _save_snapshot(report, label=snap_label)
                    st.success(f"Saved: `{saved_path.name}`")
                except Exception as e:
                    st.error(f"Save failed: {e}")

        st.markdown("<hr style='margin:1rem 0;border-color:#E2E8F0'>", unsafe_allow_html=True)
        st.markdown("#### 📂 Load Previous Analysis")

        snapshots = _list_snapshots()
        if not snapshots:
            st.info("No saved snapshots yet. Save the current analysis above.")
        else:
            for snap_path in snapshots:
                mtime = datetime.fromtimestamp(snap_path.stat().st_mtime)
                size  = snap_path.stat().st_size // 1024

                s_col1, s_col2, s_col3 = st.columns([4, 1, 1])
                with s_col1:
                    st.markdown(
                        f"<div style='font-size:0.85rem;color:#1A2B3C;font-weight:500'>"
                        f"{snap_path.stem}</div>"
                        f"<div style='font-size:0.72rem;color:#94A3B8'>"
                        f"{mtime.strftime('%d %b %Y %H:%M')} · {size} KB</div>",
                        unsafe_allow_html=True
                    )
                with s_col2:
                    if st.button("Load", key=f"load_{snap_path.name}",
                                 use_container_width=True):
                        try:
                            loaded = _load_snapshot(snap_path)
                            # normalize_report is in app.py — call from there
                            st.session_state.report_data       = loaded
                            st.session_state.analysis_done     = True
                            st.session_state.analysis_source   = "snapshot"
                            st.session_state.selected_item_idx = None
                            st.session_state.page = "results"
                            st.rerun()
                        except Exception as e:
                            st.error(f"Load failed: {e}")
                with s_col3:
                    if st.button("🗑️", key=f"del_{snap_path.name}",
                                 help="Delete this snapshot",
                                 use_container_width=True):
                        snap_path.unlink(missing_ok=True)
                        st.rerun()

    # ── TAB 3: Print View ─────────────────────────────────────────────────
    with tab_print:
        st.markdown("#### 🖨️ Print-Friendly Summary")
        st.markdown(
            "Optimised for screenshots and stakeholder presentations. "
            "Use your browser's print function (Ctrl+P) after expanding this view."
        )

        # Collect per-regulation stats for print view
        reg_stats: dict[str, dict] = {}
        for item in items:
            src = Path(
                item.get("regulation_source") or item.get("source") or "Unknown"
            ).stem
            cls = item.get("classification", "unmatched")
            if src not in reg_stats:
                reg_stats[src] = {"aligned": 0, "partial": 0, "gap": 0, "unmatched": 0, "total": 0}
            if cls in reg_stats[src]:
                reg_stats[src][cls] += 1
            reg_stats[src]["total"] += 1

        # Top 5 gaps by risk
        top_gaps = sorted(
            [i for i in items if i.get("classification") in ("gap", "unmatched")],
            key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(
                (x.get("llm_explanation") or {}).get("risk_level", "low").lower(), 3
            )
        )[:5]

        print_html = f"""
        <div style="font-family:'DM Sans',sans-serif;max-width:900px;margin:0 auto;
                    padding:2rem;background:white;border:1px solid #E2E8F0;border-radius:12px">

            <div style="border-bottom:2px solid #0D1B2A;padding-bottom:1rem;margin-bottom:1.5rem">
                <h2 style="margin:0;font-family:'DM Serif Display',serif;color:#0D1B2A">
                    REG-INSIGHT — Compliance Report
                </h2>
                <div style="font-size:0.8rem;color:#94A3B8;margin-top:0.25rem">
                    Generated: {datetime.now().strftime('%d %B %Y %H:%M')}
                    &nbsp;·&nbsp; Hash: <code>{h}</code>
                </div>
            </div>

            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:1.5rem">
                <div style="text-align:center;padding:1rem;background:#E8F5EE;border-radius:8px">
                    <div style="font-size:1.8rem;font-weight:700;color:#1E7E55">{aligned}</div>
                    <div style="font-size:0.72rem;color:#64748B;text-transform:uppercase">Aligned</div>
                </div>
                <div style="text-align:center;padding:1rem;background:#FDF3E0;border-radius:8px">
                    <div style="font-size:1.8rem;font-weight:700;color:#D4820A">{partial}</div>
                    <div style="font-size:0.72rem;color:#64748B;text-transform:uppercase">Partial</div>
                </div>
                <div style="text-align:center;padding:1rem;background:#FDECEA;border-radius:8px">
                    <div style="font-size:1.8rem;font-weight:700;color:#C0392B">{gap}</div>
                    <div style="font-size:0.72rem;color:#64748B;text-transform:uppercase">Gaps</div>
                </div>
                <div style="text-align:center;padding:1rem;background:#F1F5F9;border-radius:8px">
                    <div style="font-size:1.8rem;font-weight:700;color:#475569">{coverage}%</div>
                    <div style="font-size:0.72rem;color:#64748B;text-transform:uppercase">Coverage</div>
                </div>
            </div>

            <h3 style="color:#0D1B2A;font-size:1rem;margin-bottom:0.75rem">
                Coverage by Regulation
            </h3>
            <table style="width:100%;border-collapse:collapse;font-size:0.82rem;margin-bottom:1.5rem">
                <thead>
                    <tr style="background:#F8FAFC">
                        <th style="text-align:left;padding:0.5rem;border:1px solid #E2E8F0">Regulation</th>
                        <th style="text-align:center;padding:0.5rem;border:1px solid #E2E8F0;color:#1E7E55">Aligned</th>
                        <th style="text-align:center;padding:0.5rem;border:1px solid #E2E8F0;color:#D4820A">Partial</th>
                        <th style="text-align:center;padding:0.5rem;border:1px solid #E2E8F0;color:#C0392B">Gap</th>
                        <th style="text-align:center;padding:0.5rem;border:1px solid #E2E8F0">Total</th>
                        <th style="text-align:center;padding:0.5rem;border:1px solid #E2E8F0">Coverage</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(
                        f'<tr>'
                        f'<td style="padding:0.45rem 0.5rem;border:1px solid #E2E8F0">{reg}</td>'
                        f'<td style="text-align:center;padding:0.45rem;border:1px solid #E2E8F0">{counts["aligned"]}</td>'
                        f'<td style="text-align:center;padding:0.45rem;border:1px solid #E2E8F0">{counts["partial"]}</td>'
                        f'<td style="text-align:center;padding:0.45rem;border:1px solid #E2E8F0">{counts["gap"]}</td>'
                        f'<td style="text-align:center;padding:0.45rem;border:1px solid #E2E8F0">{counts["total"]}</td>'
                        f'<td style="text-align:center;padding:0.45rem;border:1px solid #E2E8F0">'
                        f'{round((counts["aligned"]+counts["partial"])/counts["total"]*100,1) if counts["total"] else 0}%</td>'
                        f'</tr>'
                        for reg, counts in sorted(reg_stats.items())
                    )}
                </tbody>
            </table>

            <h3 style="color:#0D1B2A;font-size:1rem;margin-bottom:0.75rem">
                Top 5 Priority Gaps
            </h3>
            {"".join(
                f'<div style="border:1px solid #FECACA;background:#FEF2F2;'
                f'border-radius:8px;padding:0.85rem 1rem;margin-bottom:0.6rem">'
                f'<div style="font-weight:600;font-size:0.82rem;color:#991B1B;margin-bottom:0.3rem">'
                f'#{i+1} · {(item.get("llm_explanation") or {}).get("risk_level","").upper()} RISK'
                f'</div>'
                f'<div style="font-size:0.8rem;color:#374151;margin-bottom:0.3rem">'
                f'{item.get("regulation_text","")[:180]}…</div>'
                f'<div style="font-size:0.75rem;color:#6B7280">'
                f'Score: {float(item.get("final_score") or item.get("best_score") or 0):.3f}'
                f'</div></div>'
                for i, item in enumerate(top_gaps)
            )}

            <div style="margin-top:1.5rem;padding-top:1rem;border-top:1px solid #E2E8F0;
                        font-size:0.72rem;color:#94A3B8;text-align:center">
                Generated by REG-INSIGHT · Report Hash: {h} ·
                For compliance review purposes only
            </div>
        </div>
        """
        st.markdown(print_html, unsafe_allow_html=True)