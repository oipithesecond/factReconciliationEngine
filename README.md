# Fact Knowledge Layer

A multi-agent retrieval-augmented generation (RAG) system that extracts, grounds, and dynamically reconciles facts across multiple PDFs. 

Rather than treating extraction as a basic text-to-JSON pipe, this system utilizes a multi-step reasoning pipeline to ensure facts are contextually accurate, strictly traceable to exact quotes, and logically cross-referenced against historical knowledge.

## Setup and Run Instructions

This project is designed to be evaluated immediately. The local `chroma_db` vector database containing the processed vectors from the starter PDFs is pre-committed to this repository, meaning **you do not need to wait for the initial documents to process.**

**1. Clone and Install Dependencies:**
```bash
git clone <repo-url>
cd <repo-name>
pip install -r requirements.txt
```

**2. API Configuration:**
Create a `.env` file in the root directory. The system is designed to seamlessly auto-detect your provider based on the key you supply, making it incredibly easy for you to test.

To ensure a frictionless review process for you without credit card requirements, you can use a free Groq API key (OpenAI compatible):
```env
# .env
GROQ_API_KEY=your_api_key_here
```
*(Alternatively, simply provide an `OPENAI_API_KEY` instead, and the system will automatically configure itself to use OpenAI models without requiring any code changes).*

**3. Run the Dashboard:**
```bash
python -m streamlit run app.py   
```

## Video Demo

**[[YouTube Demo Link](https://github.com/oipithesecond/factReconciliationEngine)]**

*(The 3-minute demo covers the live processing of a new PDF and walks through the four required cases: Corroboration, Contradiction, Contextual Reconciliation, and Failure Handling).*

## Approach

The architecture is built on Python, Streamlit, and a persistent local ChromaDB instance. To solve the notorious unreliability of basic PDF parsing and LLM extraction, I implemented several advanced mechanisms:

### 1. Extraction Pipeline & Data Grounding
* **Page-Level Context Summarization (Signal vs. Noise):** Financial PDFs are full of noise (e.g., Tables of Contents, disclaimers, glossary terms). The system is strictly instructed to define a "fact" as a meaningful quantitative metric, historical data point, or definitive business/economic statement, and expressly skip structural pages. Before extracting these facts, the Extractor LLM is forced to generate a brief "Page Analysis" summarizing what the page means. Facts are then extracted against this context to ensure they are high-signal, drastically reducing hallucinations.
* **Overlapping Context Chunking (The Sliding Window):** To prevent facts from being cut in half at page breaks, the extractor passes the last 500 characters of the *previous* page as context when analyzing the *current* page.
* **Table-to-Markdown Injection:** Raw PDF text extraction destroys table layouts. The script uses `pdfplumber` to explicitly identify tabular grids, converts them into clean Markdown tables, and injects them alongside the text. This allows the LLM to cleanly understand dense financial metrics.
* **Strict Traceability:** Every single extracted fact is required to return the **exact substring quote** and **page number**, ensuring 0% ungrounded hallucinations.

### 2. Multi-Agent Reasoning Engine
Instead of relying on one massive, expensive prompt, the reconciliation logic is split into a "separation of concerns" to save tokens and improve reasoning:
* **The Skeptic Agent:** Acts purely as a devil's advocate. It queries ChromaDB and aggressively flags *anything* that looks like a potential conflict between a new fact and historical facts.
* **The Reconciler Agent:** Only invoked if the Skeptic flags an issue. It acts as the judge, analyzing time, scope, and unit contexts to categorize the relationship as a genuine *Contradiction* or successfully *Reconciled by Context*.

### 3. Production & Rate-Limit Awareness
* **Free-Tier Resilience:** To ensure the app can run on free-tier API accounts (which have strict RPM/TPM limits), the UI includes configurable sliders for **"Max Pages to Process"** (for testing) and **"API Delay between facts"**. This gives you complete control to throttle requests and prevent `429 RateLimit` crashes when batch-processing large PDFs.

## Limitations and Next Steps

**Limitations:**
* **Processing Speed:** While Groq is incredibly fast, processing 100+ page PDFs sequentially still takes time due to the dense table extraction and two-step reasoning per fact. 
* **Complex Multi-Table Context:** While the Table-to-Markdown injection works beautifully for standard tables, highly complex, nested financial tables spanning multiple pages can occasionally confuse the strict JSON schema.

**Next Steps (What I would build next):**
* **Async & Batch Processing:** I would transition the chunk processing and LLM calls to `asyncio` to process pages in parallel, drastically cutting down upload wait times.
* **Graph Database Integration:** While ChromaDB handles the semantic search well, linking the finalized Corroborated/Contradictory facts into a Neo4j knowledge graph would allow for much deeper "multi-hop" reasoning across entities (e.g., tracking a specific subsidiary's revenue across 5 years).
* **Dynamic Schema Evolution:** Allowing the LLM to propose new "Topics" dynamically and cluster similar topics in the UI to prevent metadata fragmentation.

## Additional Notes

* **Graceful Failure Handling:** In instances where the LLM fails to output valid JSON (e.g., dense Table of Contents pages) or hallucinates a quote, the system catches the `JSONDecodeError`. Instead of crashing, it flags the instance as a "Failure" in the UI (Case #4) and gracefully continues processing the rest of the document.
* **Real-Time UI Logging:** To solve the UX issue of Streamlit "hanging" during long processes, the UI features real-time terminal-style logging, allowing you to watch the chunking, extracting, and reasoning agents work in real-time.
