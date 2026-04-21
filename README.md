#  REG-INSIGHT

AI-powered regulatory compliance analysis system.

**An AI-powered system for automated regulatory compliance gap detection in the FinTech sector.**

REG-INSIGHT transforms the slow, manual, and error-prone process of compliance auditing into a fast, automated, and insightful workflow. It ingests raw regulatory and policy documents (PDFs) and produces a detailed, actionable report highlighting specific gaps in compliance.


##  The Problem

Compliance officers in heavily regulated industries like FinTech spend hundreds of hours manually reading dense legal documents and cross-referencing them against internal company policies. This process is:
- **Slow:** Takes weeks or months to complete.
- **Expensive:** Requires significant man-hours from highly-paid experts.
- **Error-Prone:** Susceptible to human fatigue and oversight.
- **Static:** The report is outdated as soon as new regulations are published.

REG-INSIGHT is designed to be an intelligent assistant that automates this heavy lifting, allowing compliance teams to focus on remediation rather than discovery.

##  Key Features

- **Automated PDF Ingestion:** Intelligently extracts and cleans text from both regulation and policy PDFs using `PyMuPDF`.
- **Intelligent Chunking & Filtering:** Goes beyond simple chunking by using an `ObligationExtractor` to filter out non-actionable text (definitions, examples) and focus only on testable rules.
- **Multi-Stage Semantic Analysis:**
  - **Fast Retrieval (Bi-Encoder):** Uses `sentence-transformers` for a fast semantic search to find relevant policy candidates.
  - **Precision Scoring (Cross-Encoder):** Re-ranks the top candidates with a more powerful cross-encoder for highly accurate relevance scoring.
- **LLM-Powered Compliance Verification:** The core innovation, the `CoverageChecker`, uses a Groq-powered Llama-3 model to determine if a policy *actually fulfills the intent* of a regulation, moving beyond simple text similarity.
- **AI-Generated Gap Analysis:** For every identified gap, the system generates:
  - A clear, human-readable **summary** of the compliance issue.
  - An assessed **risk level** (High, Medium, or Low).
  - A specific, **actionable recommendation** to fix the policy.
- **Interactive Dashboard:** A user-friendly `Streamlit` application to explore results, with charts, filters, and a side-by-side text comparison view.
- **Exportable Reports:** Download findings as CSV, JSON, or a formatted Markdown report for easy sharing.
- **Systematic Quality Control:** Includes a `PromptOptimizer` and `QualityAnalyzer` to A/B test prompts and automatically score the quality of the AI's output, ensuring reliability.

##  System Architecture

The project is built as a multi-layered pipeline, where each layer performs a specific task. This modular design is inspired by the structure in `verify_backend.py`.

1.  **Layer 1: Ingestion & Chunking**
    - `PDFLoader` extracts and cleans text from PDFs.
    - `DocumentChunker` splits text into meaningful paragraphs.
    - `ObligationExtractor` uses Regex and an LLM to filter for actual, testable rules.

2.  **Layer 2: Vectorization & Storage**
    - `EmbeddingGenerator` (`all-MiniLM-L6-v2`) converts text chunks into vector embeddings.
    - `ChromaManager` stores these embeddings in a persistent `ChromaDB` vector database.

3.  **Layer 3: Candidate Generation**
    - `CandidateGenerator` performs a fast semantic search (bi-encoder) on ChromaDB to retrieve the top N most relevant policy chunks for each regulation.
    - A `PolicyRouter` ensures regulations are only compared against relevant policy documents (e.g., DPDP Act vs. Data Protection Policy).

4.  **Layer 4: Precision Scoring & Classification**
    - `CrossEncoderScorer` re-ranks the candidates for high accuracy.
    - `GapClassifier` uses the final score to classify the relationship as `Aligned`, `Partial`, `Gap`, or `Unmatched`.

5.  **Layer 5: Compliance Verification**
    - The `CoverageChecker` uses an LLM to perform a normative check, determining if the policy *truly satisfies* the obligation's intent. This is the key step that moves beyond similarity.

6.  **Layer 6: Explanation & Quality Control**
    - `RefinedExplanationGenerator` uses a Groq-powered Llama-3 model with an optimized prompt to generate the final analysis for gaps.
    - `QualityAnalyzer` automatically scores the LLM's output against a rubric for accuracy, actionability, and clarity.

7.  **Application Stack**
    - **Backend:** A `FastAPI` server wraps the analysis pipeline, allowing it to be run as a background job.
    - **Frontend:** A `Streamlit` web application (`app.py`) provides the user interface for running analysis and exploring results.

##  Tech Stack

- **Backend:** Python, FastAPI
- **Frontend:** Streamlit
- **AI / ML:**
  - `sentence-transformers`: For bi-encoder and cross-encoder models.
  - `groq`: For fast inference with the Llama-3 LLM.
  - `langchain`: For document chunking.
- **Data & Storage:**
  - `chromadb`: Vector database for semantic search.
  - `PyMuPDF (fitz)`: High-performance PDF text extraction.
  - `sqlite3`: Caching LLM responses to reduce cost and latency.
- **Visualization:** `plotly`


### 1. Running the Web Application

First, start the FastAPI backend server.
```bash
uvicorn api:app --reload --port 8000
```

In a separate terminal, run the Streamlit frontend.
```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` to use the application.

### 2. Running Analysis via CLI

You can run the full analysis pipeline directly from the command line.

```bash
# Run analysis using heuristics (fast, no LLM for explanations)
python run_compliance_analysis.py --no-llm

# Run analysis with full LLM-powered explanations
python run_compliance_analysis.py

# Generate explanations for an existing report
python run_explanations.py --generate --input outputs/your_report.json
```

### 3. Verifying the Backend

A built-in script allows you to test each layer of the backend pipeline to ensure all components are working correctly.

```bash
python verify_backend.py
```

##  Project Structure

```
REG-INSIGHT/
├── api.py                  # FastAPI backend server
├── app.py                  # Streamlit frontend application
├── config/                 # Configuration files (API keys, thresholds)
├── data/
│   ├── raw/                # Source PDF documents
│   ├── processed/          # Processed data (e.g., ChromaDB)
│   └── llm_cache.db        # Cache for LLM responses
├── outputs/                # Generated compliance reports (JSON, CSV)
├── src/                    # Core source code for the pipeline
│   ├── ingestion/          # PDF loading and chunking
│   ├── embeddings/         # Vector embedding and ChromaDB management
│   ├── candidate_generation/ # Bi-encoder search and candidate ranking
│   ├── scoring/            # Cross-encoder scoring and gap classification
│   ├── extraction/         # Obligation extraction and coverage checking
│   ├── explanation/        # LLM explanation generation
│   └── prompt_engineering/ # Prompt optimization and quality analysis
├── tests/                  # Unit and integration tests
├── verify_backend.py       # End-to-end backend verification script
└── requirements.txt        # Python dependencies
```
