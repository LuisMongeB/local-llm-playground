import os
import torch
from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModel
)

from .core import get_device


def load_model(model_id: str, **kwargs) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Loads a model and tokenizer with generic kwargs handling.

    Args:
        model_id: The Hugging Face model ID.
        **kwargs: Additional arguments passed to from_pretrained (e.g., trust_remote_code, load_in_4bit).

    Returns:
        A tuple of (model, tokenizer).
    """
    device = get_device()

    load_config = {
        "device_map": (
            "auto" if device == "cuda" else "mps"
        ),  # auto works best with CUDA/accelerate
        "dtype": (
            torch.bfloat16
            if torch.cuda.is_bf16_supported() or device == "mps"
            else torch.float16
        ),
    }

    load_config.update(kwargs)

    print(f"Loading {model_id} on {device} with config: {load_config}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)

    if device == "mps" and "device_map" not in kwargs:
        # On MPS, usually we load then move, or let accelerate handle it if supported.
        # For this simple loader, we'll instantiate and then explicitly move if not using device_map.
        # However, if 'load_in_4bit' is used, bitsandbytes might manage placement.
        pass

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_config)

    if "device_map" not in load_config and not getattr(model, "is_quantized", False):
        model.to(device)

    return model, tokenizer


def load_t5gemma(
    model_id: str = "google/t5gemma-2-4b-4b", **kwargs
) -> tuple[PreTrainedModel, AutoProcessor]:
    """
    Loads T5Gemma model and processor for text-only or multimodal tasks.

    T5Gemma is an encoder-decoder model that uses AutoProcessor instead of AutoTokenizer.
    It supports both text-only and image+text inputs.

    Args:
        model_id: The Hugging Face model ID (default: google/t5gemma-2-4b-4b).
        **kwargs: Additional arguments passed to from_pretrained (e.g., trust_remote_code, load_in_4bit).

    Returns:
        A tuple of (model, processor).

    Example:
        >>> model, processor = load_t5gemma()
        >>> # Text-only usage
        >>> inputs = processor(text="Explain quantum physics", return_tensors="pt")
        >>> outputs = model.generate(**inputs, max_new_tokens=50)
        >>> print(processor.decode(outputs[0], skip_special_tokens=True))
    """
    device = get_device()

    load_config = {
        "device_map": "auto" if device == "cuda" else None,
        "torch_dtype": (
            torch.bfloat16
            if torch.cuda.is_bf16_supported() or device == "mps"
            else torch.float16
        ),
    }

    load_config.update(kwargs)

    print(f"Loading T5Gemma {model_id} on {device} with config: {load_config}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if not hasattr(tokenizer, "image_token_id") or tokenizer.image_token_id is None:
        special_tokens = {
            "additional_special_tokens": ["<start_of_image>", "<boi>", "<eoi>"]
        }
        tokenizer.add_special_tokens(special_tokens)
        tokenizer.image_token_id = tokenizer.convert_tokens_to_ids("<start_of_image>")

    if not hasattr(tokenizer, "boi_token"):
        tokenizer.boi_token = "<boi>"
        tokenizer.boi_token_id = tokenizer.convert_tokens_to_ids("<boi>")
    if not hasattr(tokenizer, "eoi_token"):
        tokenizer.eoi_token = "<eoi>"
        tokenizer.eoi_token_id = tokenizer.convert_tokens_to_ids("<eoi>")

    processor = AutoProcessor.from_pretrained(model_id, tokenizer=tokenizer)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **load_config)

    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    if "device_map" not in load_config and not getattr(model, "is_quantized", False):
        model.to(device)

    return model, processor


def load_clara(model_id: str = "apple/CLaRa-7B-Instruct") -> tuple[PreTrainedModel, None]:
    """
    Loads Apple CLaRa model for retrieval-augmented generation.

    CLaRa is a unified RAG model that uses a custom API: generate_from_text.
    It does not require a standard tokenizer in the return tuple for the experiment loop
    because its generation method handles tokenization internally or expects different inputs.

    Args:
        model_id: The Hugging Face model ID (default: apple/CLaRa-7B-Instruct).

    Returns:
        A tuple of (model, None). The second element is None because we don't need a tokenizer/processor
        for the custom generation loop used with CLaRa.
    """
    device = get_device()

    print(f"Loading CLaRa {model_id} on {device}...")
    
    # Download/cache the specific subfolder
    # We use a local directory relative to the project root or wherever this is run
    local_dir_root = "./models/apple_clara"
    os.makedirs(local_dir_root, exist_ok=True)
    
    print(f"Downloading/checking model in {local_dir_root}...")
    local_dir = snapshot_download(
        repo_id=model_id,
        allow_patterns=["compression-16/*"],
        local_dir=local_dir_root,
    )
    
    model_path = os.path.join(local_dir, "compression-16")
    print(f"Loading from local path: {model_path}")

    # CLaRa uses AutoModel with trust_remote_code=True
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" or device == "mps" else torch.float32,
    ).to(device)

    return model, None
