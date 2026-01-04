import sys
import os
from dotenv import load_dotenv

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from llm_playground.loader import load_t5gemma
from llm_playground.evaluate import run_t5gemma_generation, grade_results
from llm_playground.judge import OpenAIJudge
from llm_playground.core import cleanup

def main():
    """
    T5Gemma 2 (4B-4B) evaluation with text-only inputs.

    This experiment tests the google/t5gemma-2-4b-4b model on our evaluation dataset.
    T5Gemma is a multimodal encoder-decoder model, but we're using text-only inputs here.
    """
    load_dotenv()

    model_id = "google/t5gemma-2-4b-4b"

    print("="*80)
    print(f"T5Gemma 2 Experiment: {model_id}")
    print("="*80)

    print(f"\nLoading {model_id}...")
    model, processor = load_t5gemma(model_id)

    print("\nRunning generation on evaluation dataset...")
    results = run_t5gemma_generation(model, processor, "data/eval_set.json", max_new_tokens=256)

    # Cleanup model and processor BEFORE grading to avoid fork issues
    print("\nCleaning up model resources...")
    del model, processor
    cleanup()

    # Print generated results
    print("\n" + "="*80)
    print("Generated Results:")
    print("="*80)
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Question: {result['question']}")
        print(f"    Reference: {result['reference']}")
        print(f"    Generated: {result['generated_answer']}")

    # Grade results using OpenAI Judge (after model cleanup)
    print("\n" + "="*80)
    print("Grading results with GPT-4.1-mini...")
    print("="*80)

    judge = OpenAIJudge(model_name="gpt-4.1-mini")
    graded_results = grade_results(results, judge)

    # Print scores
    print("\n" + "="*80)
    print("Final Scores:")
    print("="*80)
    for i, result in enumerate(graded_results, 1):
        print(f"[{i}] Score: {result['score']}/5 - {result['question'][:50]}...")

    print("\nT5Gemma experiment complete!")

if __name__ == "__main__":
    main()
