# Fact Reconciliation Engine

A multi-agent retrieval-augmented generation (RAG) system that extracts, grounds, and dynamically reconciles facts across multiple PDFs.

Rather than treating extraction as a basic text-to-JSON pipe, this system uses a multi-step reasoning pipeline to ensure facts are contextually accurate, strictly traceable to verified quotes, and logically cross-referenced against historical knowledge.

## Setup and Run

**1. Clone and Install Dependencies:**
```bash
git clone <repo-url>
cd factReconciliationEngine
pip install -r requirements.txt
```

**2. API Configuration:**
Copy `.env.example` to `.env` in the root directory and add an API key. Provider and model selection is handled in `config.py`, which auto-detects your provider based on the key you supply:

```env
# .env
RAPIDROUTER_API_KEY=your_api_key_here
```
*(Alternatively, you can provide a `GROQ_API_KEY` or an `OPENAI_API_KEY` instead, or override defaults using `LLM_PROVIDER` and `LLM_MODEL`.)*

**3. Run the Dashboard:**
```bash
python -m streamlit run app.py
```

Runtime files (ChromaDB vectors and extracted/dashboard JSON records) are stored locally under `data/` (gitignored).

## Architecture

The pipeline is built on Python, Streamlit, `pdfplumber`, and a persistent local ChromaDB instance:

### 1. Extraction Pipeline & Data Grounding (`extractor.py`)
* **Page-Level Context Summarization (Signal vs. Noise):** Financial and economic PDFs contain significant structural noise (Tables of Contents, disclaimers, glossaries). The extractor defines a "fact" as a meaningful quantitative metric, historical data point, or definitive business/economic statement and skips structural pages, generating a brief `page_analysis` summary before extraction.
* **Overlapping Context Chunking (Sliding Window):** To prevent facts from being split across page breaks, the extractor passes the last 500 characters of the *previous* page as context when analyzing the *current* page.
* **Table-to-Markdown Injection:** `pdfplumber` identifies tabular grids, converts them into Markdown tables, and injects them alongside the page text so tabular metrics retain their structure.
* **Programmatic Quote Verification:** Every extracted fact includes a `source_quote` and `page_number`, and `verify_source_quote()` programmatically verifies (with whitespace normalization) that the quote appears in the source text, surfacing a `Quote Verified` / `Quote Unverified` indicator in the UI.

### 2. Vector Memory & Deduplication (`memory.py`)
* **Local Embeddings & Persistence:** Uses `all-MiniLM-L6-v2` embeddings in a persistent ChromaDB store (`data/chroma_db`).
* **Deterministic Deduplication:** Facts are keyed by a SHA-256 hash of `(source_pdf, page_number, fact)` and stored via `upsert`, preventing duplicate entries when re-processing documents.
* **Lifecycle Management:** Supports deleting facts by source PDF or resetting the collection.

### 3. Multi-Agent Reasoning Engine (`reasoning.py`)
Reconciliation is split across two focused agents using structured JSON completions (`config.py`):
* **Skeptic Agent:** Queries ChromaDB for similar historical facts and acts as a devil's advocate, flagging potential conflicts or confirming alignment.
* **Reconciler Agent:** Invoked only when the Skeptic flags an issue. It compares time periods, scopes, and units to classify the relationship as **Corroborated**, **Contradiction**, or **Reconciled by Context**.

### 4. Dashboard & Processing Controls (`app.py`)
* **Concurrent Extraction & Throttling:** Supports parallel page extraction via `ThreadPoolExecutor` (configurable worker count), per-PDF page limits, and configurable API delays to avoid rate-limit errors.
* **Batched Persistence & Fault Tolerance:** Saves extracted and reconciled records once per page, and logs JSON/API failures under the **Failure** tab without halting the batch.
* **Knowledge Dashboard:** Displays KPI summary metrics, full-text search across topics/facts/quotes/reasoning, source PDF filtering, JSON/CSV export of filtered results, and Knowledge Base management (per-PDF deletion or full reset).

## Limitations and Roadmap

**Current Limitations:**
* **Sequential Per-Fact Reconciliation:** While page extraction runs concurrently, facts are reconciled sequentially against ChromaDB so each fact can reference earlier facts from the same run.
* **Complex Multi-Page Tables:** Highly nested financial tables spanning multiple pages can occasionally challenge strict JSON extraction.

**Planned Improvements:**
* **Knowledge Graph Integration:** Complement ChromaDB semantic search with a graph store (e.g., Neo4j) for multi-hop reasoning across entities and time periods.
* **Dynamic Topic Clustering:** Cluster related topics in the UI to prevent metadata fragmentation.

## License

This project is licensed under the [MIT License](LICENSE).
