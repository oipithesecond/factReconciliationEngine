import os
import json
import re
import time
import pdfplumber
from dotenv import load_dotenv
from openai import RateLimitError, InternalServerError
from pydantic import BaseModel, Field
from config import create_json_completion, get_client_and_model

# defining the JSON schema 
class Fact(BaseModel):
    topic: str = Field(description="The general subject entity or topic of the fact (e.g., a company name, a macroeconomic indicator, a product, a regulatory policy).")
    fact: str = Field(description="The extracted factual statement. Must be a meaningful quantitative metric, historical data point, or definitive business/economic statement.")
    time_period: str = Field(description="The time period this fact refers to, if mentioned (e.g., 'FY25', '2023', 'Q4'). Use 'Not specified' if absent.")
    source_quote: str = Field(description="The EXACT substring from the provided text that supports this fact.")
    page_number: int = Field(description="The page number where the fact was found.")

class FactList(BaseModel):
    page_analysis: str = Field(description="A brief analysis of the page content, determining what is signal (important financial/business data) and what is noise to ignore.")
    facts: list[Fact]

def extract_text_with_tables(page):
    """
    Extracts text and tables from a pdfplumber page.
    Formats tables as Markdown to help LLM structure.
    """
    text = page.extract_text()
    if not text:
        return ""
        
    tables = page.extract_tables()
    if tables:
        text += "\n\n[Extracted Tables in Markdown format for reference]:\n"
        for table in tables:
            valid_rows = []
            for row in table:
                # Keep rows that have at least one non-empty cell
                if row and any(cell and str(cell).strip() for cell in row):
                    cleaned_row = [str(cell).replace('\n', ' ').strip() if cell else "" for cell in row]
                    valid_rows.append(cleaned_row)
            
            if not valid_rows:
                continue
                
            headers = valid_rows[0]
            text += "|" + "|".join(headers) + "|\n"
            text += "|" + "|".join(["---"] * len(headers)) + "|\n"
            for row in valid_rows[1:]:
                if len(row) < len(headers):
                    row.extend([""] * (len(headers) - len(row)))
                text += "|" + "|".join(row) + "|\n"
            text += "\n"
            
    return text.strip()

# parsing and chunking 
def parse_and_chunk_pdf(pdf_path):
    print(f"Reading PDF: {pdf_path}...")
    chunks = []
    
    with pdfplumber.open(pdf_path) as pdf:
        previous_text = ""
        for page_num, page in enumerate(pdf.pages, start=1):
            text = extract_text_with_tables(page)
            if text and text.strip():
                # Take the last ~500 characters of the previous page to provide context
                overlap_text = previous_text[-500:] if previous_text else ""
                
                chunks.append({
                    "page_number": page_num,
                    "overlap_text": overlap_text,
                    "text": text.strip()
                })
                previous_text = text.strip()
                
    print(f"Successfully chunked into {len(chunks)} pages.")
    return chunks

def verify_source_quote(quote: str, page_text: str, overlap_text: str = "") -> bool:
    """Check whether the extracted quote appears in the source text (whitespace-normalized)."""
    if not quote or not quote.strip():
        return False
    norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
    haystack = norm(f"{overlap_text} {page_text}")
    return norm(quote) in haystack


# extractor agent
def extract_facts_from_chunk(client, chunk, model_name, max_retries=5):
    system_prompt = (
        f"You are an expert data extraction agent analyzing a document. "
        f"Your task is to extract critical quantitative metrics, historical financial data, and definitive statements about economic conditions, company performance, or business operations.\n\n"
        f"STRICT RULES:\n"
        f"1. IGNORE Tables of Contents, glossaries, disclaimers, cover pages, indices, and structural document references. These are NOT facts.\n"
        f"2. If a page contains no meaningful business, financial, or macroeconomic facts, return an empty list: {{\"facts\": []}}.\n"
        f"3. Do not extract trivial sentences. We only want impactful data points (e.g., revenue, GDP growth, strategic business changes, significant market shifts, operational metrics).\n"
        f"4. For each fact, you MUST provide the EXACT substring quote from the text.\n"
        f"5. The text is from page {chunk['page_number']}.\n\n"
        f"You MUST return ONLY a valid JSON object with exactly TWO keys:\n"
        f"1. 'page_analysis' (string): A brief Chain-of-Thought analysis (under 50 words) deciding what is important signal to extract and what is noise to ignore.\n"
        f"2. 'facts' (list): A list of objects containing the extracted facts.\n"
        f"Each object in the 'facts' list must have exactly these keys: 'topic' (string), 'fact' (string), 'time_period' (string), 'source_quote' (string), and 'page_number' (integer).\n"
        f"IMPORTANT: Output the raw JSON immediately. Do NOT wrap the JSON in markdown blocks. Do NOT add any introductory or concluding text."
    )
    
    user_prompt = f"Text to analyze:\n"
    if chunk['overlap_text']:
        user_prompt += f"[Context from previous page]:\n{chunk['overlap_text']}\n\n"
    user_prompt += f"[Current Page Text]:\n{chunk['text']}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    print(f"Analyzing page {chunk['page_number']}...")
    delay = 10
    
    for attempt in range(1, max_retries + 1):
        try:
            data = create_json_completion(
                client,
                model_name,
                messages,
                temperature=0.0,
                max_tokens=8192,  # prevents JSON truncation on dense tables
            )
            # Enforce page_number mapping and verify source quotes against page text
            if "facts" in data:
                for fact in data["facts"]:
                    fact["page_number"] = chunk["page_number"]
                    fact["quote_verified"] = verify_source_quote(
                        fact.get("source_quote", ""),
                        chunk.get("text", ""),
                        chunk.get("overlap_text", ""),
                    )
            return data
        except json.JSONDecodeError:
            print(f"\n[DEBUG] JSON Decode Error on page {chunk['page_number']} (Attempt {attempt}/{max_retries}).")
            time.sleep(2)
        except (RateLimitError, InternalServerError) as e:
            print(f"Rate limit / server busy on page {chunk['page_number']}. Retrying in {delay}s (Attempt {attempt}/{max_retries})...")
            time.sleep(delay)
            delay = min(delay * 2, 70) 
        except Exception as e:
            print(f"Unexpected error on page {chunk['page_number']}: {e}")
            time.sleep(5)
    
    raise RuntimeError(f"Exceeded max retries for page {chunk['page_number']}")

# main
if __name__ == "__main__":
    load_dotenv()

    client, model_name = get_client_and_model()
    if not client:
        print("Error: Please set RAPIDROUTER_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY in your .env file.")
        exit(1)
        
    pdf_file = "sample.pdf" 
    output_file = "data/extracted_facts.json"

    all_extracted_facts = []
    processed_pages = set()
    
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                all_extracted_facts = json.load(f)
                processed_pages = {item["page_number"] for item in all_extracted_facts}
                print(f"Found existing progress: {len(processed_pages)} pages already processed.")
        except Exception:
            all_extracted_facts = []

    document_chunks = parse_and_chunk_pdf(pdf_file)

    for chunk in document_chunks:
        p_num = chunk["page_number"]

        if p_num in processed_pages:
            continue

        try:
            result = extract_facts_from_chunk(client, chunk, model_name)
            page_facts = result.get("facts", [])
            all_extracted_facts.extend(page_facts)
            processed_pages.add(p_num)

            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_extracted_facts, f, indent=4)

            time.sleep(4)
        except Exception as e:
            print(f"Skipping page {p_num} after failure: {e}")

    print(f"\nFinished! Total extracted facts: {len(all_extracted_facts)}")
    print(f"Data saved in {output_file}")