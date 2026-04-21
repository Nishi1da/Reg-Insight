# save as add_policies.py in your project root
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────
CHROMA_PATH   = "data/processed/chroma_db"
COLLECTION    = "policies"
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 50
MODEL_NAME    = "all-MiniLM-L6-v2"

NEW_PDFS = [
    "data/policies/VDA_service_pvt_ltd.pdf",
    "data/policies/Finsecure_pvt_ltd.pdf",
    "data/policies/datasafe_pvt.pdf",
    "data/policies/Horizon_nbfc.pdf",
    "data/policies/swiftpay_soln.pdf",
]

# ── Load embedder ─────────────────────────────────────────────────
print(f"Loading embedder: {MODEL_NAME}")
embedder  = SentenceTransformer(MODEL_NAME)
splitter  = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# ── Connect to ChromaDB ───────────────────────────────────────────
client     = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"}
)

print(f"Connected to collection '{COLLECTION}'")
print(f"Current chunk count: {collection.count()}")

# ── Process each PDF ──────────────────────────────────────────────
def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text chunks from PDF with metadata."""
    doc    = fitz.open(pdf_path)
    chunks = []
    
    full_text = ""
    page_map  = []  # track which page each char belongs to
    
    for page_num, page in enumerate(doc, 1):
        page_text = page.get_text()
        page_map.append((len(full_text), len(full_text) + len(page_text), page_num))
        full_text += page_text + "\n"
    
    doc.close()
    
    # Split into chunks
    text_chunks = splitter.split_text(full_text)
    
    for i, chunk_text in enumerate(text_chunks):
        if len(chunk_text.strip()) < 20:  # skip tiny chunks
            continue
        chunks.append({
            "text":     chunk_text.strip(),
            "chunk_id": i,
        })
    
    return chunks


total_added = 0

for pdf_path in NEW_PDFS:
    path = Path(pdf_path)
    
    if not path.exists():
        print(f"\n❌ File not found: {pdf_path}")
        print(f"   Check the filename and path")
        continue
    
    print(f"\nProcessing: {path.name}")
    
    # Check if already indexed
    existing = collection.get(
        where={"source": str(path.name)},
        include=["metadatas"]
    )
    if existing["ids"]:
        print(f"  ⚠️  Already indexed ({len(existing['ids'])} chunks) — skipping")
        print(f"  To re-index, delete and re-add manually")
        continue
    
    # Extract chunks
    chunks = extract_text_from_pdf(str(path))
    print(f"  Extracted {len(chunks)} chunks")
    
    if not chunks:
        print(f"  ❌ No text extracted — PDF may be scanned/image-based")
        continue
    
    # Generate embeddings in batches
    batch_size = 32
    texts      = [c["text"] for c in chunks]
    
    print(f"  Generating embeddings...")
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embs  = embedder.encode(batch, show_progress_bar=False)
        all_embeddings.extend(embs.tolist())
    
    # Add to ChromaDB
    ids        = [f"{path.stem}_chunk_{c['chunk_id']:04d}" for c in chunks]
    metadatas  = [
        {
            "source":       path.name,
            "source_stem":  path.stem,
            "chunk_id":     c["chunk_id"],
            "collection":   COLLECTION,
        }
        for c in chunks
    ]
    
    # Add in batches
    batch_size_db = 100
    for i in range(0, len(ids), batch_size_db):
        collection.add(
            ids        = ids[i:i+batch_size_db],
            documents  = texts[i:i+batch_size_db],
            embeddings = all_embeddings[i:i+batch_size_db],
            metadatas  = metadatas[i:i+batch_size_db],
        )
    
    total_added += len(chunks)
    print(f"  ✅ Added {len(chunks)} chunks from {path.name}")

# ── Final summary ─────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"INDEXING COMPLETE")
print(f"{'='*50}")
print(f"Chunks added this run : {total_added}")
print(f"Total collection size : {collection.count()}")

# Show all sources now in collection
all_results = collection.get(include=["metadatas"])
from collections import Counter
sources = Counter(m.get("source", "unknown") for m in all_results["metadatas"])
print(f"\nAll policy sources in ChromaDB:")
for src, count in sorted(sources.items(), key=lambda x: -x[1]):
    print(f"  {count:4d}  {src}")