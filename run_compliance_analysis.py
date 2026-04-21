"""
run_compliance_analysis.py
─────────────────────────
New main pipeline for REG-INSIGHT.

Replaces pure similarity scoring with actual compliance verification.

Usage:
    # Dry run (no API calls):
    python run_compliance_analysis.py --dry-run

    # Heuristic mode (no Groq, instant results):
    python run_compliance_analysis.py --no-llm --output outputs/compliance_heuristic.json

    # Full analysis with LLM:
    python run_compliance_analysis.py --output outputs/compliance_report.json

    # Quick test on 20 chunks:
    python run_compliance_analysis.py --limit 20 --output outputs/test_report.json
"""

import sys
import json
import argparse
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def load_existing_gap_report(path: str) -> Optional[Dict]:
    """Load your existing Week 4 gap report"""
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.error(f"Failed to load report: {e}")
        return None


def run_compliance_analysis(
    gap_report_path: str,
    output_path: str,
    limit: Optional[int] = None,
    dry_run: bool = False,
    use_llm: bool = True
):
    """
    Main compliance analysis pipeline.

    Takes your existing gap report as input and enriches each item
    with proper compliance verification instead of just similarity scores.
    """

    print("=" * 70)
    print("REG-INSIGHT: Compliance Gap Analysis")
    print("=" * 70)

    # ── Load existing report ──────────────────────────────────────────────
    print(f"\nLoading report: {gap_report_path}")
    report = load_existing_gap_report(gap_report_path)

    if report is None:
        print(f"ERROR: Could not load {gap_report_path}")
        print("Run your Week 4 gap analyzer first:")
        print("  python src/scoring/gap_analyzer.py -a -o outputs/gap_report.json")
        sys.exit(1)

    # Extract items from report
    items = []
    if isinstance(report, list):
        items = report
    elif isinstance(report, dict):
        for key in ("regulation_analysis", "results", "gaps",
                    "items", "analysis", "gap_analysis"):
            if key in report and isinstance(report[key], list):
                items = report[key]
                print(f"Found items under key: '{key}'")
                break

    if not items:
        print("ERROR: Could not find gap items in report")
        print("Top-level keys:", list(report.keys()) if isinstance(report, dict) else "LIST")
        sys.exit(1)

    if limit:
        items = items[:limit]

    print(f"Found {len(items)} items to analyze")

    # ── Initialize components ─────────────────────────────────────────────
    from src.retrieval.policy_router import PolicyRouter
    from src.extraction.obligation_extractor import ObligationExtractor
    from src.extraction.coverage_checker import CoverageChecker

    router = PolicyRouter()

    # Initialize Groq if not dry run and LLM requested
    groq_client = None
    if use_llm and not dry_run:
        try:
            from src.explanation.groq_client import GroqLLMClient
            groq_client = GroqLLMClient()
            health = groq_client.health_check()
            if health['status'] == 'healthy':
                print(f"✅ Groq API connected ({health['latency_ms']:.0f}ms)")
                usage = groq_client.check_daily_usage()
                print(f"   Daily usage: {usage['used_today']}/{usage['daily_limit']}")
            else:
                print("⚠️  Groq unavailable, using heuristic mode")
                groq_client = None
        except Exception as e:
            print(f"⚠️  Groq init failed: {e}, using heuristic mode")
            groq_client = None
    else:
        if dry_run:
            print("ℹ️  Dry run mode — no API calls")
        else:
            print("ℹ️  Heuristic mode — no Groq API calls")

    extractor = ObligationExtractor(groq_client=groq_client)
    checker = CoverageChecker(groq_client=groq_client)

    # ── Connect to ChromaDB ───────────────────────────────────────────────
    chroma = None
    policy_lookup = {}
    try:
        from src.embeddings.chroma_manager import ChromaManager
        chroma = ChromaManager()
        # Build lookup from POLICIES collection: chunk_id -> (text, metadata)
        pol_collection = chroma.get_collection("policies")
        pol_data = pol_collection.get()
        policy_lookup = {
            chunk_id: (text, meta)
            for chunk_id, text, meta in zip(
                pol_data['ids'],
                pol_data['documents'],
                pol_data['metadatas']
            )
        }
        print(f"✅ ChromaDB connected ({len(policy_lookup)} policy chunks)")
    except Exception as e:
        print(f"⚠️  ChromaDB unavailable: {e}")

    # ── Initialize embedder for fresh policy fetching ─────────────────────
    embedder = None
    try:
        from src.embeddings.embedding_generator import EmbeddingGenerator
        embedder = EmbeddingGenerator()
        print("✅ Embedder ready")
    except Exception as e:
        print(f"⚠️  Embedder unavailable: {e}")

    # ── Print existing report summary ─────────────────────────────────────
    status_counts = defaultdict(int)
    for item in items:
        status = item.get("status") or item.get("classification", "unknown")
        status_counts[str(status).lower()] += 1

    print(f"\n{'='*70}")
    print("EXISTING REPORT SUMMARY")
    print(f"{'='*70}")
    print(f"  Total items      : {len(items)}")
    for status, count in sorted(status_counts.items()):
        print(f"  {status:20}  : {count}")

    # ── DRY RUN ───────────────────────────────────────────────────────────
    if dry_run:
        skip_count = 0
        process_count = 0
        for item in items:
            reg_text = item.get("regulation_text", item.get("chunk_text", ""))
            if not reg_text or not reg_text.strip():
                skip_count += 1
                continue
            result = extractor.process_chunk(reg_text, use_llm=False)
            if result['should_skip']:
                skip_count += 1
            else:
                process_count += 1

        print(f"\n{'='*70}")
        print("DRY RUN ANALYSIS")
        print(f"{'='*70}")
        print(f"  Would skip (definitions/examples) : {skip_count}")
        print(f"  Would process (real obligations)  : {process_count}")
        print(f"  Estimated API calls               : {process_count if use_llm else 0}")
        est_time = process_count * 3
        print(f"  Estimated time (LLM mode)         : ~{est_time}s ({est_time//60}min)")
        print(f"  Estimated time (heuristic mode)   : ~{process_count // 10}s")
        print(f"\n  Run with --no-llm for instant heuristic results")
        print(f"  Run without --dry-run for full LLM analysis")
        print(f"{'='*70}")
        return

    # ── MAIN PROCESSING LOOP ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("PROCESSING")
    print(f"{'='*70}")

    results = []
    skipped = []
    compliance_counts = defaultdict(int)
    by_regulation = defaultdict(lambda: defaultdict(int))
    start_time = time.time()

    for i, item in enumerate(items):

        # ── Read fields from item ──
        reg_text = item.get("regulation_text", item.get("chunk_text", ""))
        reg_source = item.get(
            "source",
            item.get("regulation", item.get("source_regulation", ""))
        )
        reg_id = item.get(
            "regulation_chunk_id",
            item.get("chunk_id", f"chunk_{i}")
        )
        pol_text = item.get("policy_text", "")
        pol_source = item.get(
            "policy_document",
            item.get("matched_policy", "")
        )
        pol_id = item.get("policy_chunk_id", "")

        # ── Skip empty regulation text ──
        if not reg_text or not reg_text.strip():
            skipped.append({
                **item,
                "compliance_status": "not_applicable",
                "skip_reason": "Empty regulation text"
            })
            continue

        # ── Filter definitions/examples (fast regex, no API) ──
        extraction = extractor.process_chunk(
            reg_text, reg_source, use_llm=False
        )

        if extraction['should_skip']:
            skipped.append({
                **item,
                "compliance_status": "not_applicable",
                "skip_reason": extraction['skip_reason'],
                "category": router.get_category_label(reg_source, reg_text)
            })
            print(f"  {i+1:3}. [SKIP   ] {reg_text[:60]}...")
            continue

        # ── Get routing and category ──
        relevant_policies = router.get_relevant_policies(reg_text, reg_source)
        category = router.get_category_label(reg_source, reg_text)

        # ── Fetch policy text if missing ──
        # First try: use policy_chunk_id to look up in policies collection
        if not pol_text and pol_id and pol_id in policy_lookup:
            pol_text, pol_meta = policy_lookup[pol_id]
            pol_source = pol_meta.get("source", pol_source)

        # Second try: use router + embedder to fetch fresh relevant policy
        if not pol_text and embedder is not None and chroma is not None:
            try:
                reg_embedding = embedder.encode(reg_text).tolist()
                fresh_results = router.get_policy_chunks_from_chroma(
                    reg_text, reg_source, chroma, reg_embedding, top_k=3
                )
                if (fresh_results
                        and fresh_results.get('documents')
                        and fresh_results['documents'][0]):
                    pol_text = fresh_results['documents'][0][0]
                    if fresh_results.get('metadatas') and fresh_results['metadatas'][0]:
                        pol_source = fresh_results['metadatas'][0][0].get(
                            'source', pol_source
                        )
                    if fresh_results.get('ids') and fresh_results['ids'][0]:
                        pol_id = fresh_results['ids'][0][0]
            except Exception as e:
                logger.debug(f"Fresh policy fetch failed for chunk {i}: {e}")

        # ── Check coverage ──
        if pol_text:
            old_score = item.get("best_score", item.get("score", 0.0))
            coverage = checker.check_coverage(
                regulation_text=reg_text,
                policy_text=pol_text,
                regulation_chunk_id=reg_id,
                policy_chunk_id=pol_id,
                regulation_source=reg_source,
                policy_source=pol_source,
                similarity_score=old_score,
                delay_seconds=8.0 if groq_client else 0.0
            )
            compliance_score = coverage.compliance_score
            coverage_type = coverage.coverage_type.value
            what_missing = coverage.what_is_missing
            what_covered = coverage.what_policy_covers
            reasoning = coverage.reasoning
        else:
            # Truly no policy found even after router fetch
            compliance_score = 0.0
            coverage_type = "none"
            what_missing = "No relevant policy text found"
            what_covered = ""
            reasoning = "No policy document covers this regulation category"

        # ── Classify compliance status ──
        if coverage_type == "explicit" and compliance_score >= 0.9:
            status = "compliant"
        elif coverage_type in ["explicit", "implicit"] and compliance_score >= 0.6:
            status = "substantially_compliant"
        elif coverage_type == "partial" or compliance_score >= 0.3:
            status = "partially_compliant"
        else:
            status = "non_compliant"

        compliance_counts[status] += 1
        by_regulation[category][status] += 1

        # ── Build enriched result ──
        enriched = {
            **item,
            "compliance_status": status,
            "compliance_score": round(compliance_score, 3),
            "coverage_type": coverage_type,
            "what_policy_covers": what_covered,
            "what_is_missing": what_missing,
            "compliance_reasoning": reasoning,
            "category": category,
            "relevant_policies": relevant_policies,
            "obligations": extraction.get("obligations", []),
            "analyzed_at": datetime.now().isoformat()
        }
        results.append(enriched)

        # ── Print progress line ──
        icons = {
            "compliant": "✅",
            "substantially_compliant": "🟡",
            "partially_compliant": "⚠️ ",
            "non_compliant": "❌"
        }
        icon = icons.get(status, "?")
        missing_preview = (what_missing or "")[:35] if what_missing else ""
        print(
            f"  {i+1:3}. [{icon}] "
            f"{reg_text[:42]:<42} | {missing_preview}"
        )

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    total_processed = len(results)
    total_skipped = len(skipped)

    print(f"\n{'='*70}")
    print("COMPLIANCE ANALYSIS RESULTS")
    print(f"{'='*70}")
    print(f"\n  Items processed  : {total_processed}")
    print(f"  Items skipped    : {total_skipped} (definitions, examples, empty)")
    print(f"  Time elapsed     : {elapsed:.1f}s")

    print(f"\n  COMPLIANCE BREAKDOWN:")
    for status in ["compliant", "substantially_compliant",
                   "partially_compliant", "non_compliant"]:
        count = compliance_counts[status]
        pct = count / total_processed * 100 if total_processed > 0 else 0
        bar = "█" * int(pct / 5)
        print(f"  {status:28} : {count:4} ({pct:5.1f}%) {bar}")

    print(f"\n  BY REGULATION:")
    for reg_cat, statuses in sorted(by_regulation.items()):
        total_reg = sum(statuses.values())
        covered = (
            statuses.get("compliant", 0)
            + statuses.get("substantially_compliant", 0)
        )
        pct = covered / total_reg * 100 if total_reg > 0 else 0
        print(f"  {reg_cat:35} : {covered}/{total_reg} covered ({pct:.0f}%)")

    # ── Save output ───────────────────────────────────────────────────────
    output = {
        "generated_at": datetime.now().isoformat(),
        "report_version": "2.0_compliance_verified",
        "summary": {
            "total_processed": total_processed,
            "total_skipped": total_skipped,
            "compliance_breakdown": dict(compliance_counts),
            "by_regulation": {
                k: dict(v) for k, v in by_regulation.items()
            },
            "extractor_stats": extractor.get_stats(),
            "checker_stats": checker.get_stats(),
            "router_stats": router.get_stats()
        },
        "results": results,
        "skipped": skipped
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ Report saved: {output_path}")
    print(f"{'='*70}")


# ── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="REG-INSIGHT: Compliance Gap Analysis"
    )
    parser.add_argument(
        "--input", "-i",
        default="outputs/thesis_gap_analysis9_enriched.json",
        help="Path to existing gap report JSON"
    )
    parser.add_argument(
        "--output", "-o",
        default="outputs/compliance_report_v2.json",
        help="Output path for new compliance report"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Process only first N items (for testing)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without API calls"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Use heuristics only (no Groq API calls)"
    )

    args = parser.parse_args()

    run_compliance_analysis(
        gap_report_path=args.input,
        output_path=args.output,
        limit=args.limit,
        dry_run=args.dry_run,
        use_llm=not args.no_llm
    )