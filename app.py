import csv
import io
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from dotenv import load_dotenv

from config import get_client_and_model as _get_client_and_model
from extractor import extract_facts_from_chunk, parse_and_chunk_pdf
from memory import FactMemory
from reasoning import process_fact_with_agents

load_dotenv()

st.set_page_config(page_title="Fact Reconciliation Engine", layout="wide")

DASHBOARD_DATA_FILE = "data/dashboard_data.json"
EXTRACTED_FACTS_FILE = "data/extracted_facts.json"
CATEGORIES = ["Corroborated", "Contradiction", "Reconciled by Context", "Failure", "New Fact"]


def load_json_list(path: str) -> list:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_json_list(path: str, data: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def get_client_and_model():
    client, model_name = _get_client_and_model()
    if not client:
        st.error("Please set RAPIDROUTER_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY in .env")
        st.stop()
    return client, model_name


def records_to_csv(items: list[dict]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["status", "source_pdf", "page_number", "topic", "fact", "time_period", "quote_verified", "source_quote", "reasoning"])
    for item in items:
        f = item.get("fact", {})
        writer.writerow([
            item.get("status", ""),
            item.get("source_pdf", ""),
            f.get("page_number", ""),
            f.get("topic", ""),
            f.get("fact", ""),
            f.get("time_period", ""),
            f.get("quote_verified", ""),
            f.get("source_quote", ""),
            item.get("reasoning", ""),
        ])
    return output.getvalue()


def render_fact_card(item: dict):
    fact_data = item.get("fact", {})
    source_pdf = item.get("source_pdf", "Unknown")
    status = item.get("status", "Unknown")
    reasoning = item.get("reasoning", "")
    retrieved_facts = item.get("retrieved_facts", [])

    st.markdown(f"**Topic:** {fact_data.get('topic', 'N/A')}")
    st.markdown(f"**Fact:** {fact_data.get('fact', 'N/A')}")
    st.markdown(f"**Time Period:** {fact_data.get('time_period', 'N/A')}")

    with st.expander("View Traceability & Reasoning"):
        verified = fact_data.get("quote_verified")
        badge = "" if verified is None else (" — Quote Verified" if verified else " — Quote Unverified")
        st.markdown(f"**Source Document:** {source_pdf} (Page {fact_data.get('page_number', 'N/A')}){badge}")
        st.info(f"**Source Quote:**\n> {fact_data.get('source_quote', 'N/A')}")

        st.markdown("**Agent Reasoning:**")
        if not reasoning:
            st.write("No specific reasoning provided.")
        elif "Skeptic flagged:" in reasoning and "Reconciler verdict:" in reasoning:
            parts = reasoning.split("Reconciler verdict:")
            skeptic_part = parts[0].replace("Skeptic flagged:", "").strip()
            reconciler_part = parts[1].strip()

            st.warning(f"Skeptic Flagged:\n\n{skeptic_part}")
            if status == "Reconciled by Context":
                st.success(f"Reconciler Verdict (Reconciled by Context):\n\n{reconciler_part}")
            elif status == "Contradiction":
                st.error(f"Reconciler Verdict (Contradiction):\n\n{reconciler_part}")
            else:
                st.info(f"Reconciler Verdict:\n\n{reconciler_part}")
        elif "Skeptic confirmed alignment:" in reasoning:
            skeptic_part = reasoning.replace("Skeptic confirmed alignment:", "").strip()
            st.success(f"Skeptic Confirmed Alignment:\n\n{skeptic_part}")
        elif "Skeptic found facts to be unrelated:" in reasoning:
            skeptic_part = reasoning.replace("Skeptic found facts to be unrelated:", "").strip()
            st.info(f"Skeptic Found Facts to be Unrelated:\n\n{skeptic_part}")
        elif "Skeptic flagged issue:" in reasoning and "Reconciler failed:" in reasoning:
            parts = reasoning.split("| Reconciler failed:")
            skeptic_part = parts[0].replace("Skeptic flagged issue:", "").strip()
            reconciler_part = parts[1].strip()
            st.warning(f"Skeptic Flagged:\n\n{skeptic_part}")
            st.error(f"Reconciler Failed:\n\n{reconciler_part}")
        else:
            st.write(reasoning)

        if retrieved_facts:
            st.markdown("**Historical Facts Compared:**")
            for idx, rf in enumerate(retrieved_facts):
                st.markdown(f"**{idx + 1}. {rf.get('topic', '')}**")
                st.write(f"- Fact: {rf.get('fact', '')}")
                st.write(f"- Time: {rf.get('time_period', '')}")
                st.write(f"- Source: {rf.get('source_pdf', '')} (Page {rf.get('page_number', '')})")


def render_upload_tab(client, model_name: str, memory: FactMemory):
    st.header("Upload New Document")
    uploaded_files = st.file_uploader(
        "Upload PDFs to extract and reconcile facts",
        type=["pdf"],
        accept_multiple_files=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        max_pages = st.number_input("Max Pages to Process per PDF (0 for all)", min_value=0, max_value=1000, value=3)
    with col2:
        delay_seconds = st.number_input(
            "API Delay between facts (seconds)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.5,
            help="Increase if you hit rate limits; decrease on paid plans.",
        )
    with col3:
        concurrency = st.slider(
            "Parallel Page Extraction Workers",
            min_value=1,
            max_value=5,
            value=2,
            help="Number of pages extracted concurrently. Keep at 1 for strict rate-limit tiers.",
        )

    if uploaded_files and st.button("Process Documents"):
        dashboard_data = load_json_list(DASHBOARD_DATA_FILE)
        raw_extracted_facts = load_json_list(EXTRACTED_FACTS_FILE)
        log_box = st.empty()
        log_text = ""

        def log_update(msg: str):
            nonlocal log_text
            log_text += msg + "\n"
            log_box.code(log_text, language="text")

        for uploaded_file in uploaded_files:
            temp_pdf_path = f"temp_{uuid.uuid4().hex}_{uploaded_file.name}"
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                log_update(f"Parsing and chunking {uploaded_file.name}...")
                chunks = parse_and_chunk_pdf(temp_pdf_path)

                if max_pages > 0:
                    chunks = chunks[:max_pages]
                    log_update(f"Chunked and limited to first {len(chunks)} pages.")
                else:
                    log_update(f"Chunked into {len(chunks)} pages.")

                # Extract pages concurrently, then reconcile in page order
                extracted_by_page = {}
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    future_to_page = {
                        pool.submit(extract_facts_from_chunk, client, chunk, model_name): chunk["page_number"]
                        for chunk in chunks
                    }
                    for future in as_completed(future_to_page):
                        p_num = future_to_page[future]
                        try:
                            extracted_by_page[p_num] = future.result()
                            log_update(f"Extracted page {p_num}.")
                        except Exception as e:
                            log_update(f"Error extracting page {p_num}: {e}")

                for chunk in chunks:
                    p_num = chunk["page_number"]
                    extracted = extracted_by_page.get(p_num)
                    if not extracted:
                        continue

                    facts_list = extracted.get("facts", [])
                    if not facts_list:
                        log_update(f"No facts found on page {p_num}.")
                        continue

                    raw_extracted_facts.extend(facts_list)
                    page_records = []
                    for fact in facts_list:
                        agent_result = process_fact_with_agents(client, model_name, fact, memory)
                        page_records.append({
                            "fact": fact,
                            "source_pdf": uploaded_file.name,
                            "status": agent_result.get("status"),
                            "reasoning": agent_result.get("reasoning"),
                            "retrieved_facts": agent_result.get("retrieved_facts", []),
                        })
                        memory.store_facts([fact], source_pdf=uploaded_file.name)
                        if delay_seconds > 0:
                            time.sleep(delay_seconds)

                    # Batch disk writes once per page instead of per fact
                    dashboard_data.extend(page_records)
                    save_json_list(EXTRACTED_FACTS_FILE, raw_extracted_facts)
                    save_json_list(DASHBOARD_DATA_FILE, dashboard_data)
                    log_update(f"Reconciled & saved {len(page_records)} facts from page {p_num}.")
            finally:
                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)

        log_update("All documents processed successfully!")
        st.success("Processing complete! Check the Knowledge Dashboard.")


def render_dashboard_tab(memory: FactMemory):
    st.header("Knowledge Dashboard")
    dashboard_data = load_json_list(DASHBOARD_DATA_FILE)

    if not dashboard_data:
        st.info("No facts processed yet. Upload a document in the 'Upload & Process' tab.")
        return

    # KPI Summary Metrics
    counts = {cat: sum(1 for item in dashboard_data if item.get("status") == cat) for cat in CATEGORIES}
    metric_cols = st.columns(len(CATEGORIES) + 1)
    metric_cols[0].metric("Total Facts", len(dashboard_data))
    for idx, cat in enumerate(CATEGORIES, start=1):
        metric_cols[idx].metric(cat, counts[cat])

    # Search & Source PDF Filter
    all_sources = sorted({item.get("source_pdf", "Unknown") for item in dashboard_data})
    filter_col1, filter_col2 = st.columns([2, 1])
    with filter_col1:
        search_query = st.text_input("Search facts, topics, quotes, or reasoning", placeholder="e.g., GDP, inflation, FY25...")
    with filter_col2:
        selected_sources = st.multiselect("Filter by Source PDF", options=all_sources, default=all_sources)

    filtered_data = dashboard_data
    if selected_sources:
        filtered_data = [item for item in filtered_data if item.get("source_pdf", "Unknown") in selected_sources]
    else:
        filtered_data = []

    if search_query.strip():
        q = search_query.strip().lower()
        filtered_data = [
            item
            for item in filtered_data
            if q in item.get("fact", {}).get("topic", "").lower()
            or q in item.get("fact", {}).get("fact", "").lower()
            or q in item.get("fact", {}).get("source_quote", "").lower()
            or q in item.get("reasoning", "").lower()
        ]

    # Export & Knowledge Base Management
    exp_col1, exp_col2, _ = st.columns([1, 1, 4])
    with exp_col1:
        st.download_button(
            "Export Filtered (JSON)",
            data=json.dumps(filtered_data, indent=2),
            file_name="reconciled_facts.json",
            mime="application/json",
            disabled=not filtered_data,
        )
    with exp_col2:
        st.download_button(
            "Export Filtered (CSV)",
            data=records_to_csv(filtered_data),
            file_name="reconciled_facts.csv",
            mime="text/csv",
            disabled=not filtered_data,
        )

    with st.expander("Manage Knowledge Base"):
        mgmt_col1, mgmt_col2 = st.columns(2)
        with mgmt_col1:
            pdf_to_delete = st.selectbox("Delete facts for a specific PDF", options=all_sources)
            if st.button("Delete Selected PDF Facts", type="secondary"):
                memory.delete_by_source(pdf_to_delete)
                remaining = [item for item in dashboard_data if item.get("source_pdf") != pdf_to_delete]
                save_json_list(DASHBOARD_DATA_FILE, remaining)
                st.success(f"Removed facts for {pdf_to_delete}.")
                st.rerun()
        with mgmt_col2:
            confirm_reset = st.checkbox("Confirm full reset of ChromaDB & dashboard data")
            if st.button("Clear All Knowledge Base", type="primary", disabled=not confirm_reset):
                memory.clear_all()
                save_json_list(DASHBOARD_DATA_FILE, [])
                save_json_list(EXTRACTED_FACTS_FILE, [])
                st.success("Knowledge base cleared.")
                st.rerun()

    # Categorized Fact Tabs
    tabs = st.tabs(CATEGORIES)
    for tab, category in zip(tabs, CATEGORIES):
        with tab:
            items = [item for item in filtered_data if item.get("status") == category]
            if not items:
                st.write(f"No matching facts in the '{category}' category.")
            else:
                for item in items:
                    with st.container():
                        render_fact_card(item)
                        st.divider()


def main():
    st.title("Fact Reconciliation Engine")
    client, model_name = get_client_and_model()
    memory = FactMemory()

    tab1, tab2 = st.tabs(["Upload & Process", "Knowledge Dashboard"])
    with tab1:
        render_upload_tab(client, model_name, memory)
    with tab2:
        render_dashboard_tab(memory)


if __name__ == "__main__":
    main()
