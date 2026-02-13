"""Week 1 Test Suite """

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.chunker import DocumentChunker
from src.ingestion.document_pipeline import DocumentPipeline


def test_pdf_loader():
    print("\nTEST 1: PDF Loader")
    print("=" * 50)
    
    loader = PDFLoader()
    sample_dir = Path("data/sample")
    pdf_files = list(sample_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("⚠️ No test PDFs")
        return False
    
    passed = 0
    for pdf in pdf_files:
        try:
            pages = loader.load(str(pdf))
            assert len(pages) > 0
            print(f"✅ {pdf.name}: {len(pages)} pages")
            passed += 1
        except Exception as e:
            print(f"❌ {pdf.name}: {e}")
    
    return passed == len(pdf_files)


def test_chunker():
    print("\nTEST 2: Document Chunker")
    print("=" * 50)
    
    loader = PDFLoader()
    chunker = DocumentChunker()
    
    sample_dir = Path("data/sample")
    pdf_files = list(sample_dir.glob("*.pdf"))
    
    if not pdf_files:
        return False
    
    pages = loader.load(str(pdf_files[0]))
    chunks = chunker.chunk_document(pages)
    
    print(f"✅ Created {len(chunks)} chunks")
    print(f"✅ Average size: {sum(c['metadata']['chunk_size'] for c in chunks) / len(chunks):.0f}")
    
    return len(chunks) > 0


def test_pipeline():
    print("\nTEST 3: Complete Pipeline")
    print("=" * 50)
    
    sample_dir = Path("data/sample")
    pdf_files = list(sample_dir.glob("*.pdf"))
    
    if not pdf_files:
        return False
    
    pipeline = DocumentPipeline()
    results = pipeline.process_batch([str(f) for f in pdf_files], show_progress=False)
    
    success = sum(1 for r in results if r['success'])
    total_chunks = sum(len(r['chunks']) for r in results if r['success'])
    
    print(f"✅ Processed {success}/{len(results)} files")
    print(f"✅ Total chunks: {total_chunks}")
    
    return success == len(results)


def test_provenance():
    print("\nTEST 4: Provenance Tracking")
    print("=" * 50)
    
    sample_dir = Path("data/sample")
    pdf_files = list(sample_dir.glob("*.pdf"))
    
    if not pdf_files:
        return False
    
    pipeline = DocumentPipeline()
    result = pipeline.process_file(str(pdf_files[0]))
    
    if not result['success']:
        return False
    
    chunk = result['chunks'][0]
    can_trace = (
        'chunk_id' in chunk and
        'source' in chunk and
        'page_number' in chunk
    )
    
    if can_trace:
        print(f"✅ Can trace to: {chunk['source']} page {chunk['page_number']}")
        return True
    else:
        print("❌ Cannot trace provenance")
        return False


def main():
    print("=" * 60)
    print("REG-INSIGHT Week 1: Test Suite")
    print("=" * 60)
    
    tests = [
        ("PDF Loader", test_pdf_loader),
        ("Chunker", test_chunker),
        ("Pipeline", test_pipeline),
        ("Provenance", test_provenance)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"💥 {name} crashed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n Week 1 complete!")
        return 0
    else:
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())