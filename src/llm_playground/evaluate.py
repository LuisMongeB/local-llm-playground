import json
import re
from typing import List, Dict, Any, Callable
import torch
from .core import get_device

def parse_assistant_response(text: str) -> str:
    """
    Extracts the clean assistant response from model output.

    Handles various output formats:
    - Removes "user" and "assistant" markers
    - Removes <think> tags and their content (including incomplete tags)
    - Extracts content after "assistant" keyword

    Args:
        text: Raw model output text

    Returns:
        Cleaned assistant response
    """
    assistant_match = re.search(r'assistant\s*\n(.*)', text, re.DOTALL)
    if assistant_match:
        text = assistant_match.group(1)

    text = re.sub(r'^user\s*\n.*?\n', '', text, flags=re.MULTILINE | re.DOTALL)

    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    if '<think>' in text and '</think>' not in text:
        text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)

    text = re.sub(r'\n\s*\n', '\n', text)
    text = text.strip()

    return text

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

        response = generated_text[len(prompt):] if prompt in generated_text else generated_text

        cleaned_response = parse_assistant_response(response)

        results.append({
            "question": question,
            "reference": item["reference"],
            "generated_answer": cleaned_response,
            "full_output": generated_text
        })

    return results

def run_t5gemma_generation(
    model,
    processor,
    dataset_path: str,
    max_new_tokens: int = 256
) -> List[Dict[str, Any]]:
    """
    Iterates through the dataset and generates answers using T5Gemma (text-only).

    T5Gemma is an encoder-decoder model that uses a processor instead of a tokenizer.
    This function handles the specific input format required for T5Gemma.

    Args:
        model: The T5Gemma model (from load_t5gemma).
        processor: The T5Gemma processor (from load_t5gemma).
        dataset_path: Path to the evaluation dataset JSON file.
        max_new_tokens: Maximum number of tokens to generate.

    Returns:
        List of results with question, reference, generated_answer, and full_output.
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    results = []

    print(f"Starting T5Gemma generation on {len(data)} items...")

    for item in data:
        question = item["question"]

        # T5Gemma uses processor with text parameter for text-only input
        inputs = processor(text=question, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False  # Deterministic generation for evaluation
            )

        generated_text = processor.decode(outputs[0], skip_special_tokens=True)

        results.append({
            "question": question,
            "reference": item["reference"],
            "generated_answer": generated_text.strip(),
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
