import json
from typing import List, Dict, Any, Callable
import torch
from .core import get_device

def run_generation(
    model, 
    tokenizer, 
    dataset_path: str, 
    max_new_tokens: int = 256
) -> List[Dict[str, Any]]:
    """
    Iterates through the dataset, generates answers, and returns results.
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    device = get_device()
    results = []

    print(f"Starting generation on {len(data)} items...")
    
    for item in data:
        question = item["question"]
        
        # Simple chat template application
        # If the tokenizer has a chat template, use it. Otherwise fallback to raw text.
        try:
            messages = [{"role": "user", "content": question}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # Fallback for base models or tokenizers without chat templates
            prompt = f"Question: {question}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the answer if using chat template (naive splitting)
        # Ideally, we'd use the tokenizer to decode only the new tokens, 
        # but for this smoke test, full text is fine.
        response = generated_text[len(prompt):] if prompt in generated_text else generated_text

        results.append({
            "question": question,
            "reference": item["reference"],
            "generated_answer": response.strip(),
            "full_output": generated_text
        })

    return results

def grade_results(results: List[Dict[str, Any]], judge_function: Callable[[str, str, str], int]):
    """
    Grades the generated results using a judge function.
    """
    scored_results = []
    total_score = 0
    
    for item in results:
        score = judge_function(item["question"], item["reference"], item["generated_answer"])
        item["score"] = score
        scored_results.append(item)
        total_score += score
    
    avg_score = total_score / len(results) if results else 0
    print(f"Average Score: {avg_score:.2f}/5")
    
    return scored_results
