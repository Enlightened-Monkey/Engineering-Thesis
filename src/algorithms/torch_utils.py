"""Optional PyTorch helpers (CPU/CUDA).

This project primarily uses NumPy. These helpers enable optional Torch/CUDA
implementations without adding a hard dependency on torch.
"""

from __future__ import annotations

from typing import Optional

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _require_torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for CUDA/torch backends. Install it e.g. 'pip install torch'."
        ) from exc


def resolve_device(device: Optional[str] = None, *, prefer_cuda: bool = True) -> str:
    """Resolve a torch device string.

    Args:
        device: Explicit device string like 'cpu', 'cuda', 'cuda:0'. If provided,
            it is returned as-is.
        prefer_cuda: If True and CUDA is available, returns 'cuda', else 'cpu'.
    """

    if device is not None:
        return str(device)

    torch = _require_torch()
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_device(prefer_cuda: bool = True) -> str:
    """Get the best available device (cuda or cpu).
    
    Args:
        prefer_cuda: If True and CUDA is available, returns 'cuda', else 'cpu'.
        
    Returns:
        Device string: 'cuda' or 'cpu'
    """
    return resolve_device(prefer_cuda=prefer_cuda)
