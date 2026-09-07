import os
import json
import time
import pdfplumber
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError
from pydantic import BaseModel, Field

# defining the JSON schema 
class Fact(BaseModel):
    fact: str = Field(description="The extracted factual statement.")
    source_quote: str = Field(description="The EXACT substring from the provided text that supports this fact.")
    page_number: int = Field(description="The page number where the fact was found.")

class FactList(BaseModel):
    facts: list[Fact]

# parsing and chunking 
def parse_and_chunk_pdf(pdf_path):
    """
    1 chunk = 1 page
    """
    print(f"Reading PDF: {pdf_path}...")
    chunks = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            
            # Skip empty pages (like blank pages or image-only pages)
            if text and text.strip():
                chunks.append({
                    "page_number": page_num,
                    "text": text.strip()
                })
                print(f"Chunked page {page_num}")
                
    print(f"Successfully chunked into {len(chunks)} pages.")
    return chunks

# extractor agent
def extract_facts_from_chunk(client, chunk, max_retries=5):
    """
    Sends the text chunk to the LLM and requests strict JSON back.
    """
    # extractor prompt
    prompt = f"""
    You are an expert data extraction agent.
    Extract the key facts from the following text.
    For each fact, you MUST provide the exact substring quote from the text.
    The text is from page {chunk['page_number']}.
    
    Text to analyze:
    {chunk['text']}
    """
    
    print(f"Analyzing page {chunk['page_number']}...")
    delay = 10
    
    # call the LLM
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=prompt,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': FactList,
                    'temperature': 0.0,
                },
            )
            return json.loads(response.text)

        except APIError as e:
            # slow down requests to model for rate limiting
            if e.code in [429, 503]:
                print(f"Rate limit / server busy on page {chunk['page_number']}. Retrying in {delay}s (Attempt {attempt}/{max_retries})...")
                time.sleep(delay)
                delay = min(delay * 2, 70)  # exponential backoff up to 70 seconds
            else:
                raise e
        except Exception as e:
            print(f"Unexpected error on page {chunk['page_number']}: {e}")
            time.sleep(5)
    
    raise RuntimeError(f"Exceeded max retries for page {chunk['page_number']}")

# main
if __name__ == "__main__":

    load_dotenv()

    # Ensure API key is not empty
    if "GEMINI_API_KEY" not in os.environ:
        print("Error: Please set the GEMINI_API_KEY environment variable.")
        exit(1)
        
    # initialize AI client
    client = genai.Client()
    
    # chunk the PDF
    # if name of pdf is not sample.pdf, replace 'sample.pdf' with the name of PDF file
    pdf_file = "sample.pdf" 
    output_file = "extracted_facts.json"

    all_extracted_facts = []
    processed_pages = set()
    document_chunks = parse_and_chunk_pdf(pdf_file)
    
    
    
    # process each chunk with the extractor agent
    for chunk in document_chunks:
        try:
            result = extract_facts_from_chunk(client, chunk)
            # add the facts from this page to master list
            all_extracted_facts.extend(result.get("facts", []))
        except Exception as e:
            print(f"Failed to process page {chunk['page_number']}: {e}")
            
    # save the final output to JSON 
    output_file = "extracted_facts.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_extracted_facts, f, indent=4)
        
    print(f"\nSuccess! Extracted {len(all_extracted_facts)} facts.")
    print(f"Results saved to {output_file}")