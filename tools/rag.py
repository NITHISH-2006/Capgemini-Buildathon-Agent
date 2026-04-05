"""RAG utilities for knowledge retrieval from ChromaDB."""

import os
from pathlib import Path
import chromadb

KB_DIR = Path(__file__).parents[1] / "knowledge_base"
DB_DIR = Path(__file__).parents[1] / "database" / "chroma_db"

def get_chroma_client():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(DB_DIR))

def load_faq_documents():
    faqs = []
    for faq_file in sorted(KB_DIR.glob("*.md")):
        faqs.append(faq_file.read_text(encoding="utf-8"))
    return faqs

def init_knowledge_base():
    client = get_chroma_client()
    collection = client.get_or_create_collection(name="support_kb")
    
    docs = load_faq_documents()
    if docs:
        ids = [f"faq_{i}" for i in range(len(docs))]
        # Only add if the collection is empty
        existing = collection.count()
        if existing == 0:
            collection.add(
                documents=docs,
                ids=ids
            )
            
def retrieve_context(query: str, n_results: int = 2) -> str:
    """Retrieve relevant context for a query."""
    client = get_chroma_client()
    collection = client.get_or_create_collection(name="support_kb")
    
    # Ensure initialized
    if collection.count() == 0:
        init_knowledge_base()
        
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    if results['documents'] and len(results['documents']) > 0:
        return "\n\n".join(results['documents'][0])
    return "No relevant FAQ found for the query."
