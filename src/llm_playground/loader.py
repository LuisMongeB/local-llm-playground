import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
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
    
    # Default configuration
    load_config = {
        "device_map": "auto" if device == "cuda" else None, # auto works best with CUDA/accelerate
        "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() or device == "mps" else torch.float16,
    }
    
    # Update defaults with user-provided kwargs
    load_config.update(kwargs)
    
    print(f"Loading {model_id} on {device} with config: {load_config}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    
    # Handle specific case for MPS where device_map="auto" might not be fully supported by all versions of accelerate/transformers yet
    # or requires specific handling. For simplicity in this scaffold, we rely on standard loading.
    # Note: 'device_map' logic in kwargs overrides internal logic if present.
    
    if device == "mps" and "device_map" not in kwargs:
         # On MPS, usually we load then move, or let accelerate handle it if supported.
         # For this simple loader, we'll instantiate and then explicitly move if not using device_map.
         # However, if 'load_in_4bit' is used, bitsandbytes might manage placement.
         pass

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_config)

    # Ensure model is on the correct device if not handled by device_map/accelerate
    if "device_map" not in load_config and not getattr(model, "is_quantized", False):
        model.to(device)

    return model, tokenizer
