from typing import Optional

def pick_device(device: Optional[str] = None) -> str:
    """Use the requested device, else CUDA, else Apple MPS, else CPU."""
    if device:
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"