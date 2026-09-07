import os
import json
import time
import pdfplumber
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, InternalServerError
from pydantic import BaseModel, Field

# defining the JSON schema 
class Fact(BaseModel):
    topic: str = Field(description="The general subject entity or topic of the fact (e.g., a company name, a macroeconomic indicator, a product, a regulatory policy).")
    fact: str = Field(description="The extracted factual statement. Must be a meaningful quantitative metric, historical data point, or definitive business/economic statement.")
    time_period: str = Field(description="The time period this fact refers to, if mentioned (e.g., 'FY25', '2023', 'Q4'). Use 'Not specified' if absent.")
    source_quote: str = Field(description="The EXACT substring from the provided text that supports this fact.")
    page_number: int = Field(description="The page number where the fact was found.")

class FactList(BaseModel):
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
        f"You MUST return ONLY a valid JSON object with a single key 'facts' containing a list of objects. "
        f"Each object must have exactly these keys: 'topic' (string), 'fact' (string), 'time_period' (string), 'source_quote' (string), and 'page_number' (integer).\n"
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
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
            )
            
            raw_text = response.choices[0].message.content.strip()
            
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].split("```")[0].strip()
                
            # Isolate the JSON object
            start_idx = raw_text.find('{')
            end_idx = raw_text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                raw_text = raw_text[start_idx:end_idx+1]
                
            try:
                data = json.loads(raw_text)
                # Enforce page_number mapping to prevent LLM hallucinations of page numbers
                if "facts" in data:
                    for fact in data["facts"]:
                        fact["page_number"] = chunk["page_number"]
                return data
            except json.JSONDecodeError:
                print(f"\n[DEBUG] JSON Decode Error on page {chunk['page_number']}. Model output was:\n{raw_text}\n")
                raise ValueError("Model returned invalid JSON.")

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

    # Ensure API key is not empty and initialize AI client
    if os.getenv("OPENAI_API_KEY"):
        print("using OpenAI Key")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model_name = "gpt-4o-mini"
    elif os.getenv("GROQ_API_KEY"):
        print("using GROQ Key")
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY")
        )
        model_name = "openai/gpt-oss-120b"
    else:
        print("Error: Please set either OPENAI_API_KEY or GROQ_API_KEY in your .env file.")
        exit(1)
        
    pdf_file = "sample.pdf" 
    output_file = "extracted_facts.json"

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

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_extracted_facts, f, indent=4)

            time.sleep(4)
        except Exception as e:
            print(f"Skipping page {p_num} after failure: {e}")

    print(f"\nFinished! Total extracted facts: {len(all_extracted_facts)}")
    print(f"Data saved in {output_file}")