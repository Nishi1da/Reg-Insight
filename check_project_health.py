"""
Fixed project health checker for REG-INSIGHT.

Works with BOTH output formats:
  - Week 3 pipeline: results[].candidates[].policy_text
  - Week 4 gap report: regulation_analysis[].policy_matches[].policy_text
                       or regulation_analysis[].classification
"""

import json
import random
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_classification(item: dict) -> str:
    """
    Extract classification label regardless of which pipeline produced it.

    Week 3 field: edge_case_classification  (clear_match / no_match / …)
    Week 4 field: classification             (aligned / partial / gap / unmatched)
    """
    # Week 3
    if "edge_case_classification" in item:
        return item["edge_case_classification"]

    # Week 4 → normalise to Week-3 vocabulary so the rest of the script works
    cls = item.get("classification", "unknown")
    mapping = {
        "aligned":   "clear_match",
        "partial":   "low_confidence",
        "gap":       "no_match",
        "unmatched": "no_match",
    }
    return mapping.get(cls, cls)


def _get_policy_text(item: dict) -> str:
    """
    Return the best-match policy text, no matter where it lives.

    Week 3: item["candidates"][0]["policy_text"]
    Week 4: item["policy_matches"][0]["policy_text"]  (Week-4 key)
            item["policy_matches"][0]["content"]       (alternative)
            item["policy_matches"][0]["policy_chunk_id"] (ID only — still counts as a match)
    """
    for key in ("candidates", "policy_matches"):
        matches = item.get(key)
        if matches:
            first = matches[0]
            # Try text fields first
            text = (
                first.get("policy_text")
                or first.get("content")
                or ""
            )
            if text:
                return text
            # Week 4 stores IDs + scores but no text — that's still a valid match
            # Return the policy_chunk_id as a stand-in so the checker knows a match exists
            if first.get("policy_chunk_id") or first.get("policy_source"):
                return first.get("policy_chunk_id") or first.get("policy_source") or "matched"
    # flat field (shouldn't exist but be safe)
    return item.get("policy_text", "")


def _get_score(item: dict) -> float:
    """Extract final score from either format."""
    # Week 4 nested
    scores = item.get("scores", {})
    if scores.get("final"):
        return scores["final"]
    # Week 3 / flat
    return item.get("final_score", item.get("bi_encoder_score", 0.0))


# ── main sections ─────────────────────────────────────────────────────────────

def load_data():
    candidates = [
        "outputs/gap_analysis_report.json",
        "outputs/day28_final_report.json",
        "outputs/day21_candidate_results.json",
        "outputs/week4_final_report.json",
    ]
    for p in candidates:
        path = Path(p)
        if path.exists():
            print(f"📂 Loaded: {path}")
            with open(path) as f:
                return json.load(f)

    print("❌ ERROR: No output file found.")
    print("   Tried:", candidates)
    return None


def extract_results(data: dict) -> list:
    """Return the flat list of per-regulation items."""
    for key in ("regulation_analysis", "results", "candidates"):
        val = data.get(key)
        if val and isinstance(val, list):
            return val
    return []


def system_check(results: list):
    print("\n📊 SYSTEM CHECK")
    print("=" * 50)

    total = len(results)
    candidates_found = sum(1 for r in results if _get_policy_text(r))

    print(f"Total regulations : {total}")
    print(f"With policy match : {candidates_found}")

    if total == 0:
        print("❌ ISSUE: No regulations processed")
    else:
        print("✅ Regulations extracted correctly")

    if candidates_found == 0:
        print("❌ ISSUE: No policy candidates found in results")
        print()
        print("   Likely causes:")
        print("   1. ChromaDB has no documents — run: python index_documents.py data/sample/")
        print("   2. Week 3 pipeline ran with top_k=0 or min_score too high")
        print("   3. Report was generated on an empty collection")
    else:
        print("✅ Candidate generation working")


def distribution_check(results: list):
    print("\n📊 DISTRIBUTION CHECK")
    print("=" * 50)

    counts = {
        "clear_match":    0,
        "ambiguous":      0,
        "low_confidence": 0,
        "no_match":       0,
    }

    unknown = 0
    for item in results:
        label = _get_classification(item)
        if label in counts:
            counts[label] += 1
        else:
            unknown += 1

    total = sum(counts.values())

    for k, v in counts.items():
        print(f"  {k}: {v}")
    if unknown:
        print(f"  unknown/other: {unknown}")

    print("\n📈 RATIOS:")
    for k, v in counts.items():
        ratio = v / total if total > 0 else 0
        print(f"  {k}: {ratio:.2f}")

    print("\n🧠 DIAGNOSIS:")
    if total == 0 and unknown == 0:
        print("❌ No classified results — check pipeline output format")
    elif total == 0 and unknown > 0:
        print(f"⚠️  {unknown} items have unrecognised labels — check _get_classification()")
    elif counts["clear_match"] == 0:
        print("⚠️  No clear matches → lower min_score or re-index documents")
    elif counts["no_match"] > total * 0.7:
        print("⚠️  Too many no_match → retrieval threshold may be too strict")
    else:
        print("✅ Distribution looks reasonable")


def quality_check(results: list):
    print("\n🔍 SAMPLE QUALITY CHECK")
    print("=" * 50)

    if not results:
        print("❌ No results to inspect")
        return

    sample = random.sample(results, min(3, len(results)))

    for i, item in enumerate(sample, 1):
        print(f"\n--- Example {i} ---")

        reg_text = item.get("regulation_text", "")
        print("\n📜 Regulation:")
        print(reg_text[:250] if reg_text else "(empty)")

        pol_text = _get_policy_text(item)
        if pol_text:
            print("\n📄 Policy (best match):")
            print(pol_text[:250])
            print(f"\n📊 Score : {_get_score(item)}")
            print(f"   Label : {_get_classification(item)}")
        else:
            print("\n❌ No policy match stored")


def root_cause_analysis(data: dict, results: list):
    """Extra diagnostics printed only when the system is broken."""
    print("\n🔬 ROOT CAUSE ANALYSIS")
    print("=" * 50)

    # Check top-level report keys
    print("Report keys:", list(data.keys()))

    # Check summary block
    summary = data.get("summary", {})
    if summary:
        print("Summary block:", json.dumps(summary, indent=2))

    # Show raw first item
    if results:
        print("\nFirst item (raw keys):", list(results[0].keys()))
        print("First item (truncated):")
        preview = {k: (v[:100] if isinstance(v, str) else v)
                   for k, v in results[0].items()}
        print(json.dumps(preview, indent=2, default=str))
    else:
        print("Results list is empty.")

    print("\n💡 FIX CHECKLIST:")
    print("  1. Re-index docs  : python index_documents.py data/sample/")
    print("  2. Re-run Week 3  : python src/candidate_generation/pipeline.py")
    print("  3. Re-run Week 4  : python src/scoring/gap_analyzer.py -a -l 20 -o outputs/gap_analysis_report.json")
    print("  4. Re-run this    : python check_project_health.py")


def final_verdict(results: list):
    print("\n🎯 FINAL VERDICT")
    print("=" * 50)

    total = len(results)
    candidates_found = sum(1 for r in results if _get_policy_text(r))

    if total == 0 or candidates_found == 0:
        print("❌ SYSTEM BROKEN — see ROOT CAUSE ANALYSIS above")
        return False

    counts = {"clear_match": 0, "no_match": 0}
    for r in results:
        lbl = _get_classification(r)
        if lbl == "clear_match":
            counts["clear_match"] += 1
        elif lbl == "no_match":
            counts["no_match"] += 1

    if counts["clear_match"] == 0:
        print("⚠️  SYSTEM WORKING but NEEDS TUNING (no clear matches)")
    elif counts["no_match"] > total * 0.7:
        print("⚠️  SYSTEM WORKING but TOO STRICT")
    else:
        print("✅ SYSTEM HEALTHY and WORKING")

    return True


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    data = load_data()
    if not data:
        return

    results = extract_results(data)

    system_check(results)
    distribution_check(results)
    quality_check(results)

    healthy = final_verdict(results)

    if not healthy:
        root_cause_analysis(data, results)


if __name__ == "__main__":
    main()