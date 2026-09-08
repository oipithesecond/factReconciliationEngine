import streamlit as st
import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI
from memory import FactMemory
from extractor import parse_and_chunk_pdf, extract_facts_from_chunk
from reasoning import process_fact_with_agents

# Load environment variables
load_dotenv()

# Initialize Streamlit configuration
st.set_page_config(page_title="Fact Knowledge Layer", layout="wide")

# Persistent data file for the dashboard
DASHBOARD_DATA_FILE = "dashboard_data.json"

def load_dashboard_data():
    if os.path.exists(DASHBOARD_DATA_FILE):
        try:
            with open(DASHBOARD_DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_dashboard_data(data):
    with open(DASHBOARD_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def get_client_and_model():
    if os.getenv("OPENAI_API_KEY"):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model_name = "gpt-4o-mini"
    elif os.getenv("GROQ_API_KEY"):
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY")
        )
        model_name = "openai/gpt-oss-120b"
    else:
        st.error("Please set OPENAI_API_KEY or GROQ_API_KEY in .env")
        st.stop()
    return client, model_name

def render_fact_card(item):
    fact_data = item.get("fact", {})
    source_pdf = item.get("source_pdf", "Unknown")
    status = item.get("status", "Unknown")
    reasoning = item.get("reasoning", "")
    retrieved_facts = item.get("retrieved_facts", [])
    
    st.markdown(f"**Topic:** {fact_data.get('topic', 'N/A')}")
    st.markdown(f"**Fact:** {fact_data.get('fact', 'N/A')}")
    st.markdown(f"**Time Period:** {fact_data.get('time_period', 'N/A')}")
    
    with st.expander("View Traceability & Reasoning"):
        st.markdown(f"**Source Document:** {source_pdf} (Page {fact_data.get('page_number', 'N/A')})")
        st.info(f"**Source Quote:**\n> {fact_data.get('source_quote', 'N/A')}")
        
        st.markdown(f"**Agent Reasoning:**")
        st.write(reasoning if reasoning else "No specific reasoning provided.")
        
        if retrieved_facts:
            st.markdown("**Historical Facts Compared:**")
            for idx, rf in enumerate(retrieved_facts):
                st.markdown(f"**{idx+1}. {rf.get('topic', '')}**")
                st.write(f"- Fact: {rf.get('fact', '')}")
                st.write(f"- Time: {rf.get('time_period', '')}")
                st.write(f"- Source: {rf.get('source_pdf', '')} (Page {rf.get('page_number', '')})")

def main():
    st.title("Fact Knowledge Layer")
    
    client, model_name = get_client_and_model()
    memory = FactMemory()
    
    tab1, tab2 = st.tabs(["Upload & Process", "Knowledge Dashboard"])
    
    with tab1:
        st.header("Upload New Document")
        uploaded_files = st.file_uploader("Upload PDFs to extract and reconcile facts", type=["pdf"], accept_multiple_files=True)
        
        col1, col2 = st.columns(2)
        with col1:
            max_pages = st.number_input("Max Pages to Process per PDF (0 for all)", min_value=0, max_value=1000, value=3)
        with col2:
            delay_seconds = st.number_input("API Delay between facts (seconds)", min_value=0.0, max_value=10.0, value=1.0, step=0.5, help="Increase if you hit rate limits, decrease for faster processing on paid plans.")
        
        if uploaded_files:
            if st.button("Process Documents"):
                dashboard_data = load_dashboard_data()
                
                with st.status("Starting document processing...", expanded=True) as status:
                    for uploaded_file in uploaded_files:
                        status.update(label=f"Processing {uploaded_file.name}...", state="running")
                        
                        import uuid
                        # Save temp file with UUID to prevent concurrent tab collisions
                        temp_pdf_path = f"temp_{uuid.uuid4().hex}_{uploaded_file.name}"
                        with open(temp_pdf_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        try:
                            st.write(f"Parsing and chunking {uploaded_file.name}...")
                            chunks = parse_and_chunk_pdf(temp_pdf_path)
                            
                            if max_pages > 0:
                                chunks = chunks[:max_pages]
                                st.write(f" Chunked and limited to first {len(chunks)} pages for processing.")
                            else:
                                st.write(f" Chunked into {len(chunks)} pages.")
                            
                            for i, chunk in enumerate(chunks):
                                st.write(f" Extracting facts from {uploaded_file.name} - Page {chunk['page_number']} ({i+1}/{len(chunks)})...")
                                
                                try:
                                    extracted = extract_facts_from_chunk(client, chunk, model_name)
                                    facts_list = extracted.get("facts", [])
                                    
                                    if facts_list:
                                        st.write(f"Found {len(facts_list)} facts on page {chunk['page_number']}. Reconciling...")
                                        for fact in facts_list:
                                            st.write(f"&nbsp;&nbsp;&nbsp;&nbsp; Skepticising & Reconciling: '{fact.get('topic')}'...")
                                            agent_result = process_fact_with_agents(client, model_name, fact, memory)
                                            
                                            st.write(f"&nbsp;&nbsp;&nbsp;&nbsp; Storing to DB: '{fact.get('topic')}' -> **{agent_result.get('status')}**")
                                            
                                            # Save to dashboard
                                            dashboard_record = {
                                                "fact": fact,
                                                "source_pdf": uploaded_file.name,
                                                "status": agent_result.get("status"),
                                                "reasoning": agent_result.get("reasoning"),
                                                "retrieved_facts": agent_result.get("retrieved_facts", [])
                                            }
                                            dashboard_data.append(dashboard_record)
                                            save_dashboard_data(dashboard_data)
                                            
                                            # Store fact in memory for future comparisons
                                            memory.store_facts([fact], source_pdf=uploaded_file.name)
                                            
                                            # Dynamic delay based on user input
                                            if delay_seconds > 0:
                                                time.sleep(delay_seconds) 
                                    else:
                                        st.write(f" No facts found on page {chunk['page_number']}.")
                                        
                                    # Extra small delay between pages
                                    if delay_seconds > 0:
                                        time.sleep(max(1.0, delay_seconds))

                                        
                                except Exception as e:
                                    st.error(f" Error extracting from page {chunk['page_number']}: {e}")
                                    
                        finally:
                            if os.path.exists(temp_pdf_path):
                                os.remove(temp_pdf_path)
                    
                    status.update(label="All documents processed successfully!", state="complete", expanded=False)
                st.success("Processing complete! Check the Knowledge Dashboard.")

    with tab2:
        st.header("Knowledge Dashboard")
        dashboard_data = load_dashboard_data()
        
        if not dashboard_data:
            st.info("No facts processed yet. Upload a document in the 'Upload & Process' tab.")
        else:
            categories = ["Corroborated", "Contradiction", "Reconciled by Context", "Failure", "New Fact"]
            tabs = st.tabs(categories)
            
            for tab, category in zip(tabs, categories):
                with tab:
                    # Filter items by category
                    items = [item for item in dashboard_data if item.get("status") == category]
                    
                    if not items:
                        st.write(f"No facts in the '{category}' category.")
                    else:
                        for item in items:
                            with st.container():
                                render_fact_card(item)
                                st.divider()

if __name__ == "__main__":
    main()
