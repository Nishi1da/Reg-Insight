"""Chunk Validator - Streamlit app for VS Code"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from ingestion.pdf_loader import PDFLoader
from ingestion.chunker import DocumentChunker


def main():
    st.set_page_config(
        page_title="REG-INSIGHT: Chunk Validator",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Document Chunk Validator")
    st.markdown("Inspect how PDFs are split into chunks")
    
    # Sidebar
    st.sidebar.header("Settings")
    chunk_size = st.sidebar.slider("Chunk Size", 256, 1024, 512, 64)
    chunk_overlap = st.sidebar.slider("Overlap", 0, 200, 50, 10)
    
    # File upload
    uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
    
    if uploaded_file:
        # Save temporarily
        temp_path = Path("data/processed/temp.pdf")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_bytes(uploaded_file.getvalue())
        
        with st.spinner("Processing..."):
            loader = PDFLoader()
            chunker = DocumentChunker(chunk_size, chunk_overlap)
            
            pages = loader.load(str(temp_path))
            chunks = chunker.chunk_document(pages)
            
            st.success(f" {len(pages)} pages → {len(chunks)} chunks")
            
            # Stats
            col1, col2, col3 = st.columns(3)
            col1.metric("Pages", len(pages))
            col2.metric("Chunks", len(chunks))
            avg = sum(c['metadata']['chunk_size'] for c in chunks) / len(chunks)
            col3.metric("Avg Size", f"{avg:.0f}")
            
            # Chunk browser
            st.header("Browse Chunks")
            selected = st.selectbox(
                "Select chunk",
                range(len(chunks)),
                format_func=lambda i: f"Chunk {i+1} (Page {chunks[i]['page_number']})"
            )
            
            chunk = chunks[selected]
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Metadata")
                st.json({
                    "chunk_id": chunk['chunk_id'],
                    "source": chunk['source'],
                    "page": chunk['page_number'],
                    "section": chunk['section_header'] or "(none)",
                    "chars": chunk['metadata']['chunk_size'],
                    "words": chunk['metadata']['word_count']
                })
            
            with col_b:
                st.subheader("Content")
                st.text_area("Text", chunk['content'], height=300)
            
            # Cleanup
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()