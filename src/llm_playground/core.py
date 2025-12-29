import torch
import gc

def get_device() -> str:
    """
    Detects the best available device: CUDA, MPS (Apple Silicon), or CPU.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def cleanup() -> None:
    """
    Clears memory cache for CUDA/MPS and runs garbage collection.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
