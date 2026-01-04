import os
import sys
import json
import torch
from dotenv import load_dotenv

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from llm_playground.core import cleanup
from llm_playground.evaluate import grade_results
from llm_playground.loader import load_clara


def main():
    """
    Apple CLaRa-7B-Instruct evaluation experiment.
    
    This experiment tests the apple/CLaRa-7B-Instruct model (compression-16/128)
    using its custom retrieval-augmented generation API.
    """
    load_dotenv()

    # Note: Hugging Face model ID. 
    # If the user has a specific local path or compression variant in mind, they can change this.
    model_id = "apple/CLaRa-7B-Instruct"

    print("=" * 80)
    print(f"Apple CLaRa Experiment: {model_id}")
    print("=" * 80)

    print(f"\nLoading {model_id}...")
    # load_clara returns (model, None) as tokenizer is not needed/standard
    model, _ = load_clara(model_id)
    
    dataset_path = "data/eval_set_rag.json"
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    print(f"\nRunning CLaRa generation on {len(data)} items...")
    
    results = []
    
    for item in data:
        question = item["question"]
        doc_list = item.get("documents", [])
        
        # Prepare inputs for CLaRa
        # Wrap in list as API expects batch
        questions = [question]
        
        # Documents should be a list of lists of strings (batch of document sets)
        documents = [doc_list]
        
        try:
            with torch.no_grad():
                # Call the custom API
                outputs = model.generate_from_text(
                    questions=questions,
                    documents=documents,
                    max_new_tokens=256
                )
            
            # Output handling
            generated_answer = outputs[0] if isinstance(outputs, list) else outputs
            
        except Exception as e:
            print(f"Error generating for question '{question}': {e}")
            generated_answer = f"[ERROR] {str(e)}"

        results.append({
            "question": question,
            "reference": item["reference"],
            "generated_answer": str(generated_answer).strip(),
            "full_output": str(generated_answer),
            "documents": doc_list
        })

    # Cleanup
    print("\nCleaning up model resources...")
    del model
    cleanup()

    # Print generated results
    print("\n" + "=" * 80)
    print("Generated Results:")
    print("=" * 80)
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Question: {result['question']}")
        print(f"    Reference: {result['reference']}")
        print(f"    Generated: {result['generated_answer']}")

    # Grade results
    from llm_playground.judge import OpenAIJudge

    print("\n" + "=" * 80)
    print("Grading results with GPT-4.1-mini...")
    print("=" * 80)

    judge = OpenAIJudge(model_name="gpt-4.1-mini")
    # Note: The unmodified grade_results expects [question, reference, generated_answer].
    # It won't use 'documents' for grading unless we modify the judge, but it will work for now.
    graded_results = grade_results(results, judge)

    # Print scores
    print("\n" + "=" * 80)
    print("Final Scores:")
    print("=" * 80)
    for i, result in enumerate(graded_results, 1):
        print(f"[{i}] Score: {result['score']}/5 - {result['question'][:50]}...")

    print(f"\n{model_id} experiment complete!")


if __name__ == "__main__":
    main()
