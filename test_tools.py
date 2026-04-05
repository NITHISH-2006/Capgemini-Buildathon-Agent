import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent))

from tools.rag import retrieve_context
from tools.actions import password_reset, create_ticket
from tools.sentiment import detect_sentiment

def run_tests():
    print("--- Testing RAG ---")
    try:
        ctx = retrieve_context("How do I reset my password?", n_results=1)
        print("RAG Context:", ctx[:200], "...")
    except Exception as e:
        print("RAG Error:", e)
        
    print("\n--- Testing Actions ---")
    try:
        res1 = password_reset("johndoe")
        print("password_reset output:", json.dumps(res1))
        
        res2 = create_ticket("johndoe", "My screen is blank.")
        print("create_ticket output:", json.dumps(res2))
    except Exception as e:
        print("Actions Error:", e)

    print("\n--- Testing Sentiment ---")
    try:
        s1 = detect_sentiment("I am very happy with the service!")
        print("Positive Sentiment:", json.dumps(s1))
        
        s2 = detect_sentiment("This is incredibly frustrating and I am angry.")
        print("Negative Sentiment:", json.dumps(s2))
    except Exception as e:
        print("Sentiment Error:", e)

if __name__ == "__main__":
    run_tests()
