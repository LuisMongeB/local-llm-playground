import os
import sys

from dotenv import load_dotenv

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from llm_playground.core import cleanup
from llm_playground.evaluate import grade_results, run_generation
from llm_playground.loader import load_model


def main():
    """
    Qwen3 4B Instruct evaluation experiment.

    This experiment tests the Qwen/Qwen3-4B-Instruct-2507 model on our evaluation dataset.
    This is an instruction-tuned version with better reasoning capabilities.
    """
    load_dotenv()

    model_id = "Qwen/Qwen3-4B-Instruct-2507"

    print("=" * 80)
    print(f"Qwen3 4B Instruct Model Experiment: {model_id}")
    print("=" * 80)

    print(f"\nLoading {model_id}...")
    model, tokenizer = load_model(model_id, trust_remote_code=True)

    print("\nRunning generation on evaluation dataset...")
    results = run_generation(
        model, tokenizer, "data/eval_set.json", max_new_tokens=512
    )

    # Cleanup model and tokenizer BEFORE grading to avoid fork issues
    print("\nCleaning up model resources...")
    del model, tokenizer
    cleanup()

    # Print generated results
    print("\n" + "=" * 80)
    print("Generated Results:")
    print("=" * 80)
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Question: {result['question']}")
        print(f"    Reference: {result['reference']}")
        print(f"    Generated: {result['generated_answer']}")

    # Grade results using OpenAI Judge (after model cleanup)
    # Import here to avoid loading OpenAI SDK before model cleanup
    from llm_playground.judge import OpenAIJudge

    print("\n" + "=" * 80)
    print("Grading results with GPT-4.1-mini...")
    print("=" * 80)

    judge = OpenAIJudge(model_name="gpt-4.1-mini")
    graded_results = grade_results(results, judge)

    # Print scores
    print("\n" + "=" * 80)
    print("Final Scores:")
    print("=" * 80)
    for i, result in enumerate(graded_results, 1):
        print(f"[{i}] Score: {result['score']}/5 - {result['question'][:50]}...")

    print("\nQwen3 4B Instruct experiment complete!")


if __name__ == "__main__":
    main()
