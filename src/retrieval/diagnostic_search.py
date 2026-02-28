"""
REG-INSIGHT: Retrieval Diagnostic Tool
Verifies semantic consistency, collection separation, and metadata attribution
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from embeddings.embedding_generator import EmbeddingGenerator
from embeddings.chroma_manager import ChromaManager


def flatten(results, key):
    """Flatten Chroma nested output safely"""
    data = results.get(key, [])
    return data[0] if data and isinstance(data[0], list) else []


def run_diagnostic():

    print("="*60)
    print("REG-INSIGHT: Search Diagnostic")
    print("="*60)

    embedder = EmbeddingGenerator()
    db = ChromaManager()

    collection = "regulations"

    # ----------------------------
    # Semantic Consistency Check
    # ----------------------------
    print("\n=== DIAGNOSTIC: Semantic Consistency ===")

    queries = [
        "suspicious transaction reporting",
        "STR reporting requirements",
        "reporting suspicious deals",
        "when to file STR"
    ]

    for q in queries:
        emb = embedder.encode(q)
        results = db.query([emb.tolist()], n_results=1, collection_name=collection)

        docs = flatten(results, "documents")
        metas = flatten(results, "metadatas")
        dists = flatten(results, "distances")

        source = metas[0].get("source") if metas else "UNKNOWN"
        score = 1 - dists[0] if dists else 0

        print(f"{q:<40} → {source} (score: {score:.3f})")

    # ----------------------------
    # Collection Separation
    # ----------------------------
    print("\n=== DIAGNOSTIC: Collection Separation ===")

    aml_q = embedder.encode("customer due diligence").tolist()
    dp_q = embedder.encode("data subject consent").tolist()

    aml_res = db.query([aml_q], n_results=3, collection_name=collection)
    dp_res = db.query([dp_q], n_results=3, collection_name=collection)

    aml_meta = flatten(aml_res, "metadatas")
    dp_meta = flatten(dp_res, "metadatas")

    aml_sources = [m.get("source", "UNKNOWN") for m in aml_meta]
    dp_sources = [m.get("source", "UNKNOWN") for m in dp_meta]

    print(f"AML query finds: {aml_sources}")
    print(f"DP query finds:  {dp_sources}")

    # ----------------------------
    # Source Attribution
    # ----------------------------
    print("\n=== DIAGNOSTIC: Source Attribution ===")

    test_q = embedder.encode("compliance reporting obligation").tolist()
    res = db.query([test_q], n_results=10, collection_name=collection)

    metas = flatten(res, "metadatas")

    no_meta = 0
    unknown = 0
    valid = 0

    for m in metas:
        if not m:
            no_meta += 1
        elif m.get("source") in [None, "", "unknown"]:
            unknown += 1
        else:
            valid += 1

    print(f"Results with NO metadata:    {no_meta}/{len(metas)}")
    print(f"Results with 'unknown' source: {unknown}/{len(metas)}")
    print(f"Results with valid source:   {valid}/{len(metas)}")

    if no_meta > 0:
        print("\n⚠️  Sample results missing metadata:")

    print("\n============================================================")
    print("Diagnostic Complete")
    print("============================================================")


if __name__ == "__main__":
    run_diagnostic()