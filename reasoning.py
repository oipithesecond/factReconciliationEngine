import os
import json
import time

def evaluate_fact_with_skeptic(client, model_name, new_fact, retrieved_facts, max_retries=3):
    """
    The Skeptic Agent evaluates the newly extracted fact against retrieved historical facts.
    Its sole purpose is to act as a devil's advocate and flag any genuine or likely contradictions.
    """
    if not retrieved_facts:
        return {"flagged": False, "reasoning": "No historical facts to compare against."}

    system_prompt = (
        "You are the Skeptic Agent, a critical devil's advocate. "
        "Your task is to compare a newly extracted fact with a list of historical facts. "
        "Does the new fact contradict any of the historical facts? "
        "A contradiction means the facts fundamentally disagree on a specific metric or statement for the same entity and time period, "
        "or they present mutually exclusive realities. "
        "If they are completely unrelated (e.g., different time periods, different metrics), do NOT flag it. "
        "If they agree and support each other, do NOT flag it.\n"
        "IMPORTANT: When explaining your reasoning, you MUST explicitly quote the exact source texts to justify your decision. Format your quotes as a clear, user-friendly markdown list. Do NOT use raw technical attribute names like 'source_quote', 'source_pdf', or 'page_number' in your text. Instead, write naturally and cite them cleanly (e.g., '- According to page 4 of [Document Name]: \"...\"').\n"
        "You MUST return ONLY a valid JSON object with EXACTLY these keys:\n"
        "- 'flagged' (boolean: true if there is a contradiction or likely contradiction, false otherwise)\n"
        "- 'is_corroborated' (boolean: true if at least one historical fact aligns with and supports the new fact, false if they are unrelated or contradict)\n"
        "- 'reasoning' (string: detailed explanation of your decision including cleanly formatted source quotes)."
    )
    
    clean_history = []
    for f in retrieved_facts:
        clean_history.append({
            "topic": f.get("topic"),
            "fact": f.get("fact"),
            "time_period": f.get("time_period"),
            "source_quote": f.get("source_quote"),
            "source_pdf": f.get("source_pdf"),
            "page_number": f.get("page_number")
        })

    historical_facts_str = json.dumps(clean_history, indent=2)
    new_fact_str = json.dumps({
        "topic": new_fact.get("topic"),
        "fact": new_fact.get("fact"),
        "time_period": new_fact.get("time_period"),
        "source_quote": new_fact.get("source_quote"),
        "page_number": new_fact.get("page_number")
    }, indent=2)

    user_prompt = f"Historical Facts:\n{historical_facts_str}\n\nNew Fact:\n{new_fact_str}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0
            )
            raw_text = response.choices[0].message.content.strip()
            
            # parse JSON
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].split("```")[0].strip()
                
            start_idx = raw_text.find('{')
            end_idx = raw_text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                raw_text = raw_text[start_idx:end_idx+1]
                
            data = json.loads(raw_text)
            if "flagged" not in data or "reasoning" not in data:
                raise ValueError("Missing keys in Skeptic output")
            return data
        except Exception as e:
            print(f"Skeptic Agent error (attempt {attempt+1}): {e}")
            time.sleep(2)
            
    return {"status": "Failure", "reasoning": "Skeptic Agent failed to return valid JSON after retries."}

def reconcile_fact(client, model_name, new_fact, retrieved_facts, skeptic_reasoning, max_retries=3):
    """
    The Reconciler Agent analyzes the surrounding context to classify the relationship 
    into one of three categories: Corroborated, Contradiction, Reconciled by Context.
    """
    system_prompt = (
        "You are the Reconciler Agent. The Skeptic Agent has analyzed a new fact against historical facts and flagged a potential issue. "
        "Your task is to analyze the context and classify the relationship into EXACTLY ONE of these three categories:\n"
        "1. 'Corroborated': The facts actually align and agree, even if phrased differently. The Skeptic was wrong to flag it.\n"
        "2. 'Contradiction': There is a genuine mismatch and disagreement that cannot be explained.\n"
        "3. 'Reconciled by Context': The apparent contradiction is explained by a difference in time, scope, units, or context.\n\n"
        "IMPORTANT: When explaining your verdict, especially for 'Contradiction' or 'Reconciled by Context', you MUST explicitly quote the exact source texts from both the new fact and the historical fact(s). Format these comparisons as a clear, user-friendly markdown list. Do NOT use raw technical attribute names like 'source_quote', 'source_pdf', or 'page_number' in your text. Instead, write naturally and cite them cleanly (e.g., '- Historical Fact (Page 4): \"...\"' vs '- New Fact (Page 6): \"...\"').\n"
        "You MUST return ONLY a valid JSON object with EXACTLY these keys: "
        "'status' (string: strictly one of 'Corroborated', 'Contradiction', 'Reconciled by Context') and 'reasoning' (string: detailed explanation of your final verdict including cleanly formatted source quotes)."
    )
    
    clean_history = []
    for f in retrieved_facts:
        clean_history.append({
            "topic": f.get("topic"),
            "fact": f.get("fact"),
            "time_period": f.get("time_period"),
            "source_quote": f.get("source_quote"),
            "source_pdf": f.get("source_pdf"),
            "page_number": f.get("page_number")
        })

    historical_facts_str = json.dumps(clean_history, indent=2)
    new_fact_str = json.dumps({
        "topic": new_fact.get("topic"),
        "fact": new_fact.get("fact"),
        "time_period": new_fact.get("time_period"),
        "source_quote": new_fact.get("source_quote"),
        "page_number": new_fact.get("page_number")
    }, indent=2)

    user_prompt = f"Historical Facts:\n{historical_facts_str}\n\nNew Fact:\n{new_fact_str}\n\nSkeptic's Reasoning for flagging:\n{skeptic_reasoning}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0
            )
            raw_text = response.choices[0].message.content.strip()
            
            # parse JSON
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].split("```")[0].strip()
                
            start_idx = raw_text.find('{')
            end_idx = raw_text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                raw_text = raw_text[start_idx:end_idx+1]
                
            data = json.loads(raw_text)
            if "status" not in data or "reasoning" not in data:
                raise ValueError("Missing keys in Reconciler output")
            if data["status"] not in ["Corroborated", "Contradiction", "Reconciled by Context"]:
                # Attempt to map it if it's slightly off
                status = data["status"]
                if "Corroborated" in status: data["status"] = "Corroborated"
                elif "Contradiction" in status: data["status"] = "Contradiction"
                elif "Reconciled" in status or "Context" in status: data["status"] = "Reconciled by Context"
                else: raise ValueError("Invalid status value")
                
            return data
        except Exception as e:
            print(f"Reconciler Agent error (attempt {attempt+1}): {e}")
            time.sleep(2)
            
    return {"status": "Failure", "reasoning": "Reconciler Agent failed to return valid JSON after retries."}

def process_fact_with_agents(client, model_name, new_fact, memory):
    """
    Main pipeline for Phase 3:
    1. Query memory for top 3 similar facts
    2. Run Skeptic Agent
    3. Run Reconciler Agent (if flagged)
    """
    query_text = f"{new_fact.get('topic', '')}: {new_fact.get('fact', '')}"
    
    retrieved_facts = memory.search_similar_facts(query_text, n_results=3)
    
    if not retrieved_facts:
        return {
            "status": "New Fact",
            "reasoning": "No similar historical facts found.",
            "retrieved_facts": []
        }
    
    # run Skeptic
    skeptic_result = evaluate_fact_with_skeptic(client, model_name, new_fact, retrieved_facts)
    
    if "status" in skeptic_result and skeptic_result["status"] == "Failure":
        return {
            "status": "Failure",
            "reasoning": skeptic_result.get("reasoning", "Skeptic Failed"),
            "retrieved_facts": retrieved_facts
        }
        
    flagged = skeptic_result.get("flagged", False)
    is_corroborated = skeptic_result.get("is_corroborated", False)
    skeptic_reasoning = skeptic_result.get("reasoning", "")
    
    if flagged:
        # run Reconciler
        reconciler_result = reconcile_fact(client, model_name, new_fact, retrieved_facts, skeptic_reasoning)
        
        # reconciler fails
        if reconciler_result.get("status") == "Failure":
             return {
                "status": "Failure",
                "reasoning": f"Skeptic flagged issue: {skeptic_reasoning} | Reconciler failed: {reconciler_result.get('reasoning')}",
                "retrieved_facts": retrieved_facts
            }           
            
        return {
            "status": reconciler_result.get("status"),
            "reasoning": f"Skeptic flagged: {skeptic_reasoning}\nReconciler verdict: {reconciler_result.get('reasoning')}",
            "retrieved_facts": retrieved_facts
        }
    elif is_corroborated:
        # skeptic found no contradiction and explicitly noted they align
        return {
            "status": "Corroborated",
            "reasoning": f"Skeptic confirmed alignment: {skeptic_reasoning}",
            "retrieved_facts": retrieved_facts
        }
    else:
        # not flagged, but also not corroborated (they are unrelated)
        return {
            "status": "New Fact",
            "reasoning": f"Skeptic found facts to be unrelated: {skeptic_reasoning}",
            "retrieved_facts": retrieved_facts
        }

if __name__ == "__main__":
    from dotenv import load_dotenv
    from openai import OpenAI
    from memory import FactMemory
    
    load_dotenv()
    
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
        print("Set OPENAI_API_KEY or GROQ_API_KEY")
        exit(1)
        
    memory = FactMemory()
    
    dummy_fact = {
        "topic": "Revenue",
        "fact": "The total revenue for Q1 2023 was $100 million.",
        "time_period": "Q1 2023",
        "source_quote": "Revenue reached $100M in the first quarter.",
        "page_number": 1
    }
    
    print("Testing process_fact_with_agents with a dummy fact...")
    result = process_fact_with_agents(client, model_name, dummy_fact, memory)
    print(json.dumps(result, indent=2))
