"""
Verify metadata in ChromaDB
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.embeddings.chroma_manager import ChromaManager

db = ChromaManager()
coll = db.get_collection("regulations")
data = coll.get()

print("=" * 60)
print("METADATA VERIFICATION")
print("=" * 60)
print(f"Total documents: {len(data['ids'])}")

missing_meta = 0
missing_source = 0
missing_page = 0

print("\n=== SAMPLE DOCUMENTS ===")
for i in range(min(5, len(data['ids']))):
    print(f"\nDocument {i+1}:")
    print(f"  ID: {data['ids'][i][:40]}...")
    
    if not data['metadatas'] or not data['metadatas'][i]:
        print("    NO METADATA!")
        missing_meta += 1
        continue
    
    meta = data['metadatas'][i]
    source = meta.get('source', 'MISSING')
    page = meta.get('page_number', 'MISSING')
    chunk_id = meta.get('chunk_id', 'MISSING')
    
    print(f"  Source: {source}")
    print(f"  Page: {page}")
    print(f"  Chunk ID: {chunk_id}")
    
    if source == 'MISSING' or not source:
        missing_source += 1
    if page == 'MISSING' or page is None:
        missing_page += 1

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total: {len(data['ids'])}")
print(f"Missing metadata entirely: {missing_meta}")
print(f"Missing source: {missing_source}")
print(f"Missing page: {missing_page}")
print(f"Healthy: {len(data['ids']) - missing_meta - missing_source}")

if missing_meta == 0 and missing_source == 0:
    print("\n ALL METADATA IS HEALTHY!")
else:
    print("\n  SOME METADATA IS MISSING")