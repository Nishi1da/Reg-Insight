# Project Implementation Changes

This document outlines the key deviations from the original implementation plan, detailing the changes made and the reasons behind them.

## 1. LLM Provider: Cloud API (Groq) vs. Local (Ollama)

-   **Original Plan:** Use a locally-hosted Large Language Model (LLM) via `Ollama`.
-   **Actual Implementation:** Switched to using the **Groq cloud API** with the `Llama-3` model.
-   **Reasoning:**
    -   **Performance:** The Groq API provides significantly faster inference speeds, which is critical for the real-time analysis and explanation generation features of the application. This avoids the performance bottlenecks and hardware requirements associated with running large models locally.
    -   **Scalability:** A cloud-based API is more scalable and does not require the end-user (e.g., a compliance officer) to have a powerful machine with a dedicated GPU.
    -   **Focus:** This change allowed development to focus on the core compliance logic and pipeline architecture rather than on LLM hosting and infrastructure management.

## 2. Vector Database: ChromaDB vs. FAISS

-   **Original Plan:** Evaluate both `FAISS` (a high-performance vector search library) and `ChromaDB` (a full-featured vector database).
-   **Actual Implementation:** Selected and implemented **ChromaDB** as the vector store.
-   **Reasoning:**
    -   **Developer Experience:** ChromaDB offers a simple, high-level Python API that is very intuitive for storing, managing, and querying document chunks and their embeddings.
    -   **Built-in Persistence:** It provides an out-of-the-box persistent client (`chromadb.PersistentClient`) that saves the database to disk. This was a core requirement to avoid re-processing documents on every run.
    -   **Powerful Metadata Filtering:** ChromaDB has excellent support for server-side metadata filtering within queries. This is used extensively in the project to narrow down searches (e.g., only searching within a specific policy document), a feature that is more complex to implement with a lower-level library like FAISS.
    -   **Sufficient Performance:** While FAISS is known for raw speed, ChromaDB's performance is more than sufficient for the project's scale, and its ease of use and rich feature set provided a better trade-off.

## 3. Core Logic: Compliance Verification vs. Similarity Scoring

-   **Original Plan:** Rely on semantic similarity scores (bi-encoder + cross-encoder) to classify gaps.
-   **Actual Implementation:** Introduced the **`CoverageChecker`**, a core module that uses an LLM to perform *normative compliance verification*.
-   **Reasoning:** This is the most significant architectural improvement in the project.
    -   **Superior Accuracy:** Instead of just asking "Are these texts similar?", the `CoverageChecker` asks, "Does this policy *actually fulfill the intent* of this regulation?". This dramatically reduces false positives where texts are similar but one doesn't satisfy the other (e.g., a policy defining a term vs. implementing it).
    -   **Actionable Insights:** The `CoverageChecker` provides structured, human-readable outputs, including `what_is_missing` and `what_policy_covers`. This gives compliance officers specific, actionable feedback, which is far more valuable than a simple numeric score.
    -   **Core Innovation:** This change elevates REG-INSIGHT from a sophisticated search tool to a genuine analysis engine, directly addressing the nuanced challenge of regulatory compliance.
