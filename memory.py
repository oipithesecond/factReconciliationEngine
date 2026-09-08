import os
import uuid
import chromadb
from chromadb.utils import embedding_functions

class FactMemory:
    def __init__(self, db_path="./chroma_db", collection_name="facts"):
        """
        Initializes the ChromaDB persistent client and collection.
        Uses the default embedding function (all-MiniLM-L6-v2) which runs locally.
        """
        self.db_path = db_path
        self.collection_name = collection_name
        
        # initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        
        # get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )

    def store_facts(self, facts: list[dict], source_pdf: str = "unknown"):
        """
        Store a list of facts into ChromaDB.
        Each fact should be a dictionary containing 'topic', 'fact', 'time_period', 'source_quote', 'page_number'.
        """
        if not facts:
            return

        documents = []
        metadatas = []
        ids = []

        for fact in facts:
            # create unique ID for the fact
            fact_id = str(uuid.uuid4())
            
            # the document text will be the fact itself (or combination of topic + fact)
            # this is what will be embedded and searched against.
            document = f"{fact.get('topic', '')}: {fact.get('fact', '')} (Time: {fact.get('time_period', '')})"
            
            # keep all original info in metadata so we can display it in the UI later
            metadata = {
                "topic": fact.get("topic", ""),
                "fact": fact.get("fact", ""),
                "time_period": fact.get("time_period", ""),
                "source_quote": fact.get("source_quote", ""),
                "page_number": fact.get("page_number", -1),
                "source_pdf": source_pdf
            }
            
            # ensure no None values in metadata
            metadata = {k: (v if v is not None else "") for k, v in metadata.items()}
            
            documents.append(document)
            metadatas.append(metadata)
            ids.append(fact_id)

        # Add to collection
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Stored {len(facts)} facts from {source_pdf} into memory.")

    def search_similar_facts(self, query_text: str, n_results: int = 5):
        """
        Given a new fact text, find the most semantically similar existing facts.
        """
        if self.collection.count() == 0:
            return []

        # Chroma's query method returns a dict with 'documents', 'metadatas', 'distances'
        results = self.collection.query(
            query_texts=[query_text],
            n_results=min(n_results, self.collection.count())
        )
        
        retrieved_facts = []
        if results and results.get('metadatas') and len(results['metadatas']) > 0:
            for i, metadata in enumerate(results['metadatas'][0]):
                # metadata contains all the fields we stored
                fact_info = metadata.copy()
                fact_info["distance"] = results['distances'][0][i] if 'distances' in results and results['distances'] else None
                retrieved_facts.append(fact_info)
        return retrieved_facts

if __name__ == "__main__":
    import json
    
    memory = FactMemory()
    
    # if we have extracted facts to ingest
    if os.path.exists("extracted_facts.json"):
        with open("extracted_facts.json", "r", encoding="utf-8") as f:
            facts = json.load(f)
            
        print(f"Found {len(facts)} facts in extracted_facts.json. Ingesting...")
        memory.store_facts(facts, source_pdf="sample.pdf")
        
        # Test semantic search with a sample query
        if facts:
            test_query = f"{facts[0].get('topic', '')}: {facts[0].get('fact', '')}"
            # Safely print on Windows (default encoding mismatch) 
            safe_query = test_query.encode('ascii', 'replace').decode('ascii')
            print(f"\nTesting semantic search for: '{safe_query}'")
            results = memory.search_similar_facts(test_query, n_results=3)
            print(f"Found {len(results)} similar facts:")
            for idx, res in enumerate(results):
                safe_topic = res.get('topic', '').encode('ascii', 'replace').decode('ascii')
                safe_fact = res.get('fact', '').encode('ascii', 'replace').decode('ascii')
                print(f"  {idx+1}. {safe_topic}: {safe_fact} (Distance: {res.get('distance', 'N/A')})")
    else:
        print("No extracted_facts.json found. Run the extractor first.")
