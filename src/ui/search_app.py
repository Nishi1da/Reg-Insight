"""Search App - Streamlit UI for semantic search"""

import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from retrieval.semantic_search import SemanticSearch
from embeddings.chroma_manager import ChromaManager
import time


def main():
    st.set_page_config(
        page_title="REG-INSIGHT: Semantic Search",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 REG-INSIGHT: Semantic Search")
    st.markdown("Search across regulations and policies using semantic similarity")
    
    # Initialize search
    @st.cache_resource
    def get_search_engine():
        return SemanticSearch(use_hybrid=True)
    
    try:
        search = get_search_engine()
    except Exception as e:
        st.error(f"Failed to initialize search: {e}")
        st.info("Make sure you've indexed documents first (run index_documents.py)")
        return
    
    # Sidebar filters
    st.sidebar.header("Search Settings")
    
    collection = st.sidebar.selectbox(
        "Collection",
        search.chroma_manager.list_collections(),
        index=0
    )
    
    top_k = st.sidebar.slider("Results to show", 1, 20, 5)
    use_hybrid = st.sidebar.checkbox("Hybrid search (semantic + keyword)", True)
    alpha = st.sidebar.slider("Semantic weight", 0.0, 1.0, 0.7) if use_hybrid else 0.7
    
    # Advanced filters
    with st.sidebar.expander("Advanced Filters"):
        filter_source = st.text_input("Source document (optional)")
        min_score = st.slider("Minimum relevance score", 0.0, 1.0, 0.0)
    
    # Main search
    query = st.text_input("Enter your query:", placeholder="e.g., financial reporting requirements")
    
    if query:
        # Build filters
        filters = {}
        if filter_source:
            filters["source"] = filter_source
        
        # Search
        with st.spinner("Searching..."):
            start_time = time.time()
            
            results = search.search(
                query=query,
                top_k=top_k,
                collection=collection,
                filters=filters if filters else None,
                min_score=min_score
            )
            
            latency = (time.time() - start_time) * 1000  # ms
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Results", len(results))
        col2.metric("Latency", f"{latency:.0f}ms")
        col3.metric("Collection", collection)
        
        if latency > 500:
            st.warning(" Query latency > 500ms. Consider optimization.")
        
        # Display results
        st.divider()
        
        for i, result in enumerate(results, 1):
            with st.container():
                score = result['score']
                color = "green" if score > 0.8 else "orange" if score > 0.5 else "red"
                
                st.markdown(f"### {i}. Score: :{color}[{score:.3f}]")
                
                col_a, col_b = st.columns([3, 1])
                
                with col_a:
                    st.markdown(f"**Content:**\n{result['content'][:500]}...")
                
                with col_b:
                    meta = result.get('metadata', {})
                    st.markdown("**Metadata:**")
                    st.json({
                        "Source": meta.get('source', 'Unknown'),
                        "Page": meta.get('page_number', 'N/A'),
                        "Section": meta.get('section_header', 'N/A')[:50] if meta.get('section_header') else 'N/A',
                        "Words": meta.get('word_count', 'N/A')
                    })
                    
                    # Provenance link
                    if meta.get('source') and meta.get('page_number'):
                        st.caption(f"📍 {meta['source']} p.{meta['page_number']}")
                
                # Explanation
                with st.expander("Why this result?"):
                    st.write(search.get_search_explanation(result))
                
                st.divider()
        
        # Export option
        if st.button("Export Results"):
            export_data = {
                "query": query,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": results
            }
            st.download_button(
                "Download JSON",
                data=str(export_data),
                file_name=f"search_results_{int(time.time())}.json",
                mime="application/json"
            )
    
    # Collection stats
    with st.expander("Collection Statistics"):
        try:
            count = search.chroma_manager.get_document_count(collection)
            st.write(f"Total documents: {count}")
            
            if count > 10000:
                st.info("Large collection detected (>10k docs). Performance optimized.")
        except Exception as e:
            st.error(f"Could not load stats: {e}")


if __name__ == "__main__":
    main()