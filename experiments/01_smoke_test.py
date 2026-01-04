import sys
import os
from dotenv import load_dotenv

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from llm_playground.loader import load_model
from llm_playground.evaluate import run_generation, grade_results
from llm_playground.core import cleanup

def mock_judge(question, reference, answer):
    """
    A dummy judge that just gives a random score for the smoke test.
    In a real scenario, this would call GPT-4.1 or higher.
    """
    print(f"\n[Judge]\nQ: {question}\nRef: {reference}\nGen: {answer}\n")
    return 3 # Placeholder score

def main():
    load_dotenv()
    
    # Use a tiny model for the smoke test
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    print(f"Loading {model_id}...")
    model, tokenizer = load_model(model_id)
    
    print("Running generation...")
    results = run_generation(model, tokenizer, "data/eval_set.json")
    
    print("Grading results...")
    graded = grade_results(results, mock_judge)
    
    print("Cleanup...")
    del model, tokenizer
    cleanup()
    
    print("Smoke test complete!")

if __name__ == "__main__":
    main()
