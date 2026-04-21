# REG-INSIGHT Technology Stack

This document provides a detailed breakdown of the technology stack used in the REG-INSIGHT project, categorized by function.

## Core Backend

-   **Language:** Python 3.9+
-   **Framework:** FastAPI
    -   **Purpose:** Serves the core analysis pipeline as a REST API, enabling background job processing and decoupling the frontend from the heavy computation.
    -   **Server:** Uvicorn

## Frontend

-   **Framework:** Streamlit
    -   **Purpose:** Provides an interactive, user-friendly web application for compliance officers to upload documents, run analyses, and explore results.

## AI & Machine Learning

-   **LLM Inference:** Groq API with Llama-3
    -   **Purpose:** Powers the core `CoverageChecker` for normative compliance verification and the `RefinedExplanationGenerator` for generating human-readable gap analysis, risk assessments, and recommendations. The Groq API is chosen for its extremely high-speed inference.
-   **Semantic Search & Reranking:** `sentence-transformers`
    -   **Bi-Encoder Model:** `all-MiniLM-L6-v2` is used for fast, initial candidate retrieval from the vector database.
    -   **Cross-Encoder Model:** A `ms-marco` based model is used for precise reranking of top candidates, providing highly accurate relevance scores.
-   **Document Processing:** `langchain`
    -   **Purpose:** Primarily used for its robust `RecursiveCharacterTextSplitter` to intelligently chunk large documents into meaningful paragraphs.

## Data Ingestion & Storage

-   **PDF Extraction:** `PyMuPDF (fitz)`
    -   **Purpose:** High-performance, accurate text extraction from raw PDF documents (both regulations and policies).
-   **Vector Database:** `ChromaDB`
    -   **Purpose:** Stores vector embeddings of text chunks for efficient semantic search. It is configured for persistent storage on disk.
-   **Response Caching:** `sqlite3`
    -   **Purpose:** A simple, file-based SQL database (`llm_cache.db`) used to cache responses from the Groq LLM. This significantly reduces API costs and latency on subsequent runs with the same inputs.
-   **Job Management:** `sqlite3`
    -   **Purpose:** A separate SQLite database (`jobs.db`) is used to track the status of asynchronous analysis jobs submitted via the FastAPI backend.

## Visualization & Export

-   **Charting:** `Plotly`
    -   **Purpose:** Generates interactive charts (pie charts, bar charts) for the results dashboard in the Streamlit application.
-   **Data Handling:** `pandas`
    -   **Purpose:** Used for data manipulation and creating exportable reports in formats like CSV and JSON.

## Development & Tooling

-   **Environment:** `venv`
-   **Dependencies:** `pip` and `requirements.txt`
-   **Configuration:** YAML files (`config/`) for managing API keys and pipeline thresholds.
-   **Code Quality:** The `verify_backend.py` script acts as a systematic integration test to ensure all pipeline layers are functioning correctly.