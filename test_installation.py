"""Day 1: Verify all installations"""

def test_imports():
    tests = []
    
    try:
        import fitz
        tests.append(("PyMuPDF", True))
    except ImportError as e:
        tests.append(("PyMuPDF", False))
    
    try:
        import langchain
        tests.append(("LangChain", True))
    except ImportError as e:
        tests.append(("LangChain", False))
    
    try:
        import chromadb
        tests.append(("ChromaDB", True))
    except ImportError as e:
        tests.append(("ChromaDB", False))
    
    try:
        import streamlit
        tests.append(("Streamlit", True))
    except ImportError as e:
        tests.append(("Streamlit", False))
    
    try:
        import sentence_transformers
        tests.append(("Sentence-Transformers", True))
    except ImportError as e:
        tests.append(("Sentence-Transformers", False))
    
    try:
        import yaml
        tests.append(("PyYAML", True))
    except ImportError as e:
        tests.append(("PyYAML", False))
    
    try:
        import pandas
        import numpy
        tests.append(("Pandas & NumPy", True))
    except ImportError as e:
        tests.append(("Pandas & NumPy", False))
    
    return tests

def main():
    print("=" * 60)
    print("REG-INSIGHT Day 1: Installation Verification")
    print("=" * 60)
    
    tests = test_imports()
    passed = sum(1 for _, status in tests if status)
    
    for name, status in tests:
        icon = "✅" if status else "❌"
        print(f"{icon} {name}")
    
    print("=" * 60)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All systems operational!")
    else:
        print("⚠️ Some imports failed")

if __name__ == "__main__":
    main()