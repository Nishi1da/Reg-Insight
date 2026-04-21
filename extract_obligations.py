import json
from pathlib import Path
from collections import defaultdict

with open("outputs/compliance_gap_fixed_v1.json", encoding="utf-8") as f:
    data = json.load(f)

items = data.get("results") or data.get("regulation_analysis") or data.get("items") or []

by_regulation = defaultdict(list)

for item in items:
    source = Path(item.get("regulation_source") or "unknown").stem
    status = (
        item.get("compliance_status") or 
        item.get("classification") or ""
    )
    text = (
        item.get("regulation_text") or 
        item.get("chunk_text") or ""
    ).strip()
    
    if status in ("non_compliant", "gap", "partially_compliant", "partial"):
        by_regulation[source].append(text)

for reg, chunks in sorted(by_regulation.items()):
    print(f"\n{'='*60}")
    print(f"REGULATION: {reg}")
    print(f"Gap chunks: {len(chunks)}")
    print(f"{'='*60}")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n[{i}] {chunk[:300]}")
        print("-" * 40)

output = {}
for reg, chunks in by_regulation.items():
    output[reg] = chunks

with open("outputs/gap_obligations_by_regulation.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("\n✅ Saved to outputs/gap_obligations_by_regulation.json")