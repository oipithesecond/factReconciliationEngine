# Fact Reconciliation Engine

A multi-agent retrieval-augmented generation (RAG) system that extracts, grounds, and dynamically reconciles facts across multiple PDFs.

Rather than treating extraction as a basic text-to-JSON pipe, this system uses a multi-step reasoning pipeline to ensure facts are contextually accurate, strictly traceable to exact quotes, and logically cross-referenced against historical knowledge.

## Setup and Run

**1. Clone and Install Dependencies:**
```bash
git clone <repo-url>
cd factReconciliationEngine
pip install -r requirements.txt
```

**2. API Configuration:**
Copy `.env.example` to `.env` in the root directory and add an API key. The system auto-detects your provider based on the key you supply:

```env
# .env
RAPIDROUTER_API_KEY=your_api_key_here
```
*(Alternatively, you can provide a `GROQ_API_KEY` or an `OPENAI_API_KEY` instead, and the system will automatically configure itself without requiring any code changes.)*

**3. Run the Dashboard:**
```bash
python -m streamlit run app.py
```

Runtime files (ChromaDB vectors and extracted/dashboard JSON records) are stored locally under `data/` (gitignored).

## Demo

**[YouTube Walkthrough](https://www.youtube.com/watch?v=tMikT79sOOc)**

*(Covers live PDF processing and examples of Corroboration, Contradiction, Contextual Reconciliation, and Failure Handling.)*

## Architecture

The pipeline is built on Python, Streamlit, `pdfplumber`, and a persistent local ChromaDB instance:

### 1. Extraction Pipeline & Data Grounding
* **Page-Level Context Summarization (Signal vs. Noise):** Financial and economic PDFs contain significant structural noise (Tables of Contents, disclaimers, glossaries). The extractor defines a "fact" as a meaningful quantitative metric, historical data point, or definitive business/economic statement and skips structural pages. Before extracting facts, it generates a brief `page_analysis` summary to separate signal from noise.
* **Overlapping Context Chunking (Sliding Window):** To prevent facts from being split across page breaks, the extractor passes the last 500 characters of the *previous* page as context when analyzing the *current* page.
* **Table-to-Markdown Injection:** Raw PDF text extraction breaks table layouts. `pdfplumber` identifies tabular grids, converts them into Markdown tables, and injects them alongside the page text.
* **Strict Traceability:** Every extracted fact includes an exact substring `source_quote` and `page_number`.

### 2. Multi-Agent Reasoning Engine
Reconciliation is split across two focused agents to reduce token usage and improve accuracy:
* **Skeptic Agent:** Queries ChromaDB for similar historical facts and acts as a devil's advocate, flagging anything that looks like a potential conflict or confirming alignment.
* **Reconciler Agent:** Invoked only when the Skeptic flags an issue. It compares time periods, scopes, and units to classify the relationship as **Corroborated**, **Contradiction**, or **Reconciled by Context**.

### 3. Rate-Limit Awareness & Fault Tolerance
* **Configurable Throttling:** The UI provides controls for **Max Pages to Process** and **API Delay between facts** to avoid `429 RateLimit` errors on rate-limited API plans.
* **Graceful Failure Handling:** If an LLM call fails to return valid JSON after retries, the system catches the error, logs it under the **Failure** tab in the dashboard, and continues processing the rest of the document.
* **Real-Time UI Logging:** Terminal-style logging in the Streamlit UI streams chunking, extraction, and agent progress live.

## Limitations and Roadmap

**Current Limitations:**
* **Sequential Processing:** Processing large (100+ page) PDFs sequentially takes time due to table extraction and per-fact agent calls.
* **Complex Multi-Page Tables:** Highly nested financial tables spanning multiple pages can occasionally challenge strict JSON extraction.

**Planned Improvements:**
* **Async & Batch Processing:** Transition page chunking and LLM calls to `asyncio` for concurrent processing.
* **Knowledge Graph Integration:** Complement ChromaDB semantic search with a graph store (e.g., Neo4j) for multi-hop reasoning across entities and time periods.
* **Dynamic Topic Clustering:** Cluster related topics in the UI to prevent metadata fragmentation.
