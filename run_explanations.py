"""
run_explanations.py
───────────────────
Integrates Week 5 (Groq LLM) + Week 6 (Quality Control)
with the real gap analysis report.

Fetches actual policy text from ChromaDB so the LLM sees
BOTH the regulation text AND the matched policy text.

Usage:
    python run_explanations.py                          # dry run (no API calls)
    python run_explanations.py --generate               # generate explanations
    python run_explanations.py --generate --limit 3     # test with 3 items first
"""

import sys
import json
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from prompt_engineering.refined_generator import RefinedExplanationGenerator
from scoring.gap_classifier import GapClassification, GapClass
from embeddings.chroma_manager import ChromaManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
REPORT_PATH = "outputs/thesis_gap_analysis8_enriched.json"
OUTPUT_PATH = "outputs/thesis_gap_analysis8_explained.json"
DELAY_SECONDS = 8.0
CHROMA_PATH   = "data/processed/chroma_db"


# ── Policy text fetcher ───────────────────────────────────────────────────────

class PolicyTextFetcher:
    """
    Fetches actual policy text from ChromaDB by chunk ID.

    This is the missing link — the gap report stores policy_chunk_id
    but not the actual policy text. This class retrieves it so the
    LLM can compare regulation vs policy properly.
    """

    def __init__(self, chroma_path: str = CHROMA_PATH):
        self.chroma = ChromaManager(persist_directory=chroma_path)
        self._cache: Dict[str, str] = {}
        logger.info(f"PolicyTextFetcher ready (ChromaDB: {chroma_path})")

    def get(self, policy_chunk_id: str) -> str:
        """Return policy text for a chunk ID, or empty string if not found."""
        if not policy_chunk_id:
            return ""

        if policy_chunk_id in self._cache:
            return self._cache[policy_chunk_id]

        try:
            collection = self.chroma.get_collection("policies")
            result = collection.get(ids=[policy_chunk_id])
            docs = result.get("documents", [])
            text = docs[0] if docs else ""
            self._cache[policy_chunk_id] = text
            return text
        except Exception as e:
            logger.warning(
                f"Could not fetch policy chunk {policy_chunk_id[:16]}...: {e}"
            )
            return ""

    def enrich_matches(self, policy_matches: List[Dict]) -> List[Dict]:
        """Add policy_text field to each match in a policy_matches list."""
        enriched = []
        for m in policy_matches:
            chunk_id = m.get("policy_chunk_id", "")
            text = self.get(chunk_id)
            enriched.append({
                **m,
                "policy_text": text if text else f"[See {m.get('policy_source', 'policy')}]"
            })
        return enriched


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_report(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_report(report: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved enriched report: {path}")


def build_gap_classification(entry: Dict) -> GapClassification:
    """Convert a JSON report entry into a GapClassification object."""

    cls_map = {
        "gap":       GapClass.GAP,
        "partial":   GapClass.PARTIAL,
        "aligned":   GapClass.ALIGNED,
        "unmatched": GapClass.GAP,
    }
    cls_str = entry.get("classification", "gap").lower()
    gap_class = cls_map.get(cls_str, GapClass.GAP)

    # policy_matches already enriched with policy_text by PolicyTextFetcher
    policy_matches = entry.get("policy_matches", [])

    return GapClassification(
        regulation_chunk_id = entry.get("regulation_chunk_id", ""),
        regulation_text     = entry.get("regulation_text", ""),
        regulation_metadata = {"source": entry.get("regulation_source", "")},
        classification      = gap_class,
        confidence          = entry.get("confidence", 0.5),
        confidence_level    = entry.get("confidence_level", "medium"),
        bi_encoder_score    = entry.get("bi_encoder_score", 0.0),
        cross_encoder_score = entry.get("cross_encoder_score", 0.0),
        final_score         = entry.get("final_score", 0.0),
        threshold_min       = 0.0,
        threshold_max       = 0.49,
        reasoning           = entry.get("reasoning", ""),
        recommended_action  = entry.get("recommended_action", ""),
        policy_matches      = policy_matches,
        classified_at       = datetime.now().isoformat(),
        config_version      = "v1"
    )


def print_summary(report: Dict):
    summary = report.get("summary", {})
    classifications = summary.get("classifications", {})
    gaps    = classifications.get("gap", 0)
    partial = classifications.get("partial", 0)

    print("\n" + "=" * 60)
    print("REPORT SUMMARY")
    print("=" * 60)
    print(f"  Total regulations : {summary.get('total_regulations', 0)}")
    print(f"  Coverage          : {summary.get('coverage_percentage', 0)}%")
    print(f"  Aligned           : {classifications.get('aligned', 0)}")
    print(f"  Partial           : {partial}")
    print(f"  Gaps              : {gaps}")
    print(f"  Items to explain  : {gaps + partial} API calls needed")
    print("=" * 60 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run Week 5+6 explanations on real gap report"
    )
    parser.add_argument("--generate", "-g", action="store_true",
                        help="Call Groq API and generate explanations")
    parser.add_argument("--limit", "-l", type=int, default=None,
                        help="Limit number of items (for testing)")
    parser.add_argument("--input", "-i", default=REPORT_PATH,
                        help=f"Input report (default: {REPORT_PATH})")
    parser.add_argument("--output", "-o", default=OUTPUT_PATH,
                        help=f"Output report (default: {OUTPUT_PATH})")
    args = parser.parse_args()

    # ── Load report ───────────────────────────────────────────────────────────
    print(f"\nLoading report: {args.input}")
    if not Path(args.input).exists():
        print(f"ERROR: Report not found at {args.input}")
        sys.exit(1)

    report = load_report(args.input)
    print_summary(report)

    # ── Fetch real policy text from ChromaDB ──────────────────────────────────
    print("Connecting to ChromaDB to fetch policy texts...")
    fetcher = PolicyTextFetcher(CHROMA_PATH)

    # Enrich ALL entries with real policy text (in-memory only)
    all_entries = report.get("regulation_analysis", [])
    for entry in all_entries:
        entry["policy_matches"] = fetcher.enrich_matches(
            entry.get("policy_matches", [])
        )

    # ── Filter items that need explanation ────────────────────────────────────
    to_explain = [
        e for e in all_entries
        if e.get("classification") in ("gap", "partial")
    ]

    if args.limit:
        to_explain = to_explain[:args.limit]
        print(f"Limited to {args.limit} items for testing\n")

    print(f"Items to process: {len(to_explain)}")
    for i, e in enumerate(to_explain, 1):
        # Show what policy text was found
        top_match = e["policy_matches"][0] if e["policy_matches"] else {}
        policy_preview = top_match.get("policy_text", "")[:50] or "NOT FOUND"
        print(f"  {i:2}. [{e['classification'].upper():7}] "
              f"REG: {e['regulation_text'][:50].strip()}...")
        print(f"        POL: {policy_preview}...")

    # ── Dry run ───────────────────────────────────────────────────────────────
    if not args.generate:
        print("\n" + "=" * 60)
        print("DRY RUN — no API calls made")
        print(f"Run with --generate to process {len(to_explain)} items")
        print(f"Estimated time: ~{len(to_explain) * (DELAY_SECONDS + 2):.0f}s")
        print("=" * 60)
        return

    # ── Generate explanations ─────────────────────────────────────────────────
    print("\nInitializing RefinedExplanationGenerator...")
    try:
        gen = RefinedExplanationGenerator(
            use_best_prompt=True,
            enable_quality_gate=True,
            enable_refinement=True
        )
    except Exception as e:
        print(f"ERROR: Could not initialize generator: {e}")
        sys.exit(1)

    usage = gen.groq_client.check_daily_usage()
    print(f"Groq API usage: {usage['used_today']}/{usage['daily_limit']} today")
    if usage["status"] == "critical":
        print("ERROR: Near daily API limit. Try again tomorrow.")
        sys.exit(1)

    print(f"\nGenerating {len(to_explain)} explanations...\n")

    explanations: Dict[str, Dict] = {}
    failed = 0

    for i, entry in enumerate(to_explain, 1):
        chunk_id = entry["regulation_chunk_id"]
        cls      = entry["classification"].upper()
        reg_prev = entry["regulation_text"][:55].strip()
        top_match = entry["policy_matches"][0] if entry["policy_matches"] else {}
        pol_prev = top_match.get("policy_text", "")[:55] or "NO POLICY TEXT"

        print(f"[{i:2}/{len(to_explain)}] {cls}")
        print(f"  REG: {reg_prev}...")
        print(f"  POL: {pol_prev}...")

        try:
            gap_obj = build_gap_classification(entry)
            result  = gen.generate(gap_obj)
            explanations[chunk_id] = result

            score  = result.get("quality_score", 0)
            passed = "✓" if result.get("quality_passed") else "✗"
            exp    = result.get("explanation", {})
            print(f"  {passed} quality={score:.2f} | "
                  f"risk={exp.get('risk_level', '?')} | "
                  f"summary={str(exp.get('summary',''))[:60]}...")

        except Exception as e:
            logger.error(f"  Failed: {e}")
            failed += 1

        if i < len(to_explain):
            time.sleep(DELAY_SECONDS)

    # ── Merge into report ─────────────────────────────────────────────────────
    print(f"\nMerging explanations into report...")
    enriched_count = 0

    for entry in all_entries:
        chunk_id = entry["regulation_chunk_id"]
        if chunk_id in explanations:
            exp = explanations[chunk_id]
            entry["llm_explanation"]             = exp.get("explanation", {})
            entry["explanation_quality"]         = exp.get("quality_score", 0)
            entry["explanation_quality_passed"]  = exp.get("quality_passed", False)
            entry["prompt_version"]              = exp.get("prompt_version", "")
            entry["explanation_generated_at"]    = exp.get("generated_at", "")
            enriched_count += 1

    report["report_metadata"]["explanation_run"] = {
        "generated_at":    datetime.now().isoformat(),
        "items_explained": enriched_count,
        "items_failed":    failed,
        "prompt_version":  gen.active_variant,
        "model":           gen.groq_client.model,
    }

    explanation_list = list(explanations.values())
    qa_result = gen.run_qa_validation(explanation_list)
    report["qa_summary"] = qa_result

    save_report(report, args.output)

    # ── Final metrics ─────────────────────────────────────────────────────────
    metrics = gen.get_metrics()

    # Compute quality stats directly from results (fixes cache-hit bug
    # where gen.get_metrics() returns 0.00 when all items were cached)
    all_scores = [
        v.get("quality_score", 0)
        for v in explanations.values()
        if v.get("quality_score") is not None
    ]
    all_passed = [
        v for v in explanations.values()
        if v.get("quality_passed") is True
    ]
    avg_quality   = sum(all_scores) / len(all_scores) if all_scores else 0.0
    pass_rate     = round(len(all_passed) / len(all_scores) * 100, 1) if all_scores else 0.0

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"  Explained        : {enriched_count}")
    print(f"  Failed           : {failed}")
    print(f"  Avg quality      : {avg_quality:.2f}")
    print(f"  Quality pass rate: {pass_rate}%")
    print(f"  Cache hits       : {metrics['cache_hit_rate']}%")
    print(f"  Groq calls today : {metrics['groq_usage']['used_today']}")
    print(f"  QA health        : {qa_result['overall_health']}")
    print(f"  Output saved     : {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()