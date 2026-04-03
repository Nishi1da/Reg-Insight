"""
enrich_output.py
Run from project root: python enrich_output.py

Fetches actual policy text from ChromaDB for each policy match
and adds it to the output JSON so the UI and explanations can
show regulation text AND matched policy text side by side.
"""


import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

INPUT  = "outputs/thesis_gap_analysis8.json"
OUTPUT = "outputs/thesis_gap_analysis8_enriched.json"

print("Loading ChromaDB...")
import chromadb
client     = chromadb.PersistentClient(path="data/processed/chroma_db")
pol_col    = client.get_collection("policies")
reg_col    = client.get_collection("regulations")

# Build a lookup dict: chunk_id → text for all policy chunks
print("Building policy text lookup...")
all_policies = pol_col.get(include=["documents", "metadatas"])
policy_lookup = {}
for cid, text, meta in zip(
    all_policies["ids"],
    all_policies["documents"],
    all_policies["metadatas"]
):
    policy_lookup[cid] = {
        "text":   text,
        "source": meta.get("source", ""),
        "page":   meta.get("page_number", ""),
    }

print(f"Loaded {len(policy_lookup)} policy chunks")

# Load the gap analysis output
print(f"Loading {INPUT}...")
with open(INPUT) as f:
    data = json.load(f)

items = data.get("regulation_analysis", [])
print(f"Enriching {len(items)} items...")

enriched = 0
missing  = 0

for item in items:
    matches = item.get("policy_matches", [])

    for match in matches:
        cid = match.get("policy_chunk_id", "")
        if cid in policy_lookup:
            match["policy_text"]   = policy_lookup[cid]["text"]
            match["policy_source"] = policy_lookup[cid]["source"]
            match["policy_page"]   = policy_lookup[cid]["page"]
            enriched += 1
        else:
            match["policy_text"]   = ""
            missing += 1

    # Promote best match text to top level for easy access
    if matches:
        best = max(matches, key=lambda m: m.get("final_score", 0))
        item["policy_text"]     = best.get("policy_text", "")
        item["policy_document"] = best.get("policy_source", best.get("policy_source", ""))
        item["policy_page"]     = best.get("policy_page", "")
    else:
        item["policy_text"]     = ""
        item["policy_document"] = "No match found"
        item["policy_page"]     = ""

print(f"  Enriched: {enriched} matches")
print(f"  Missing:  {missing} chunk IDs not found in ChromaDB")

# Save enriched output
with open(OUTPUT, "w") as f:
    json.dump(data, f, indent=2)

print(f"\nSaved to {OUTPUT}")
print("\nSample check — first item:")
if items:
    item = items[0]
    print(f"  Regulation : {item.get('regulation_text', '')[:150]}")
    print(f"  Policy doc : {item.get('policy_document', 'MISSING')}")
    print(f"  Policy text: {item.get('policy_text', 'MISSING')[:150]}")
    print(f"  Score      : {item.get('best_score', item.get('final_score', 'MISSING'))}")
    print(f"  Status     : {item.get('classification', 'MISSING')}")

print("\nDone. Now run:")
print("  python run_explanations.py --generate --input outputs/thesis_gap_analysis8_enriched.json")