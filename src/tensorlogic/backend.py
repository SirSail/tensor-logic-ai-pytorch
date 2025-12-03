import torch

def apply_einsum(einsum_spec: str, *tensors: torch.Tensor) -> torch.Tensor:
    """Executes an einsum operation."""
    return torch.einsum(einsum_spec, tensors)

def apply_nonlinearity(x: torch.Tensor, kind: str | None) -> torch.Tensor:
    """Applies a simple nonlinearity (relu, sigmoid, step)."""
    if kind is None:
        return x
    kind = kind.lower()
    if kind == "relu":
        return torch.relu(x)
    if kind == "sigmoid":
        return torch.sigmoid(x)
    if kind == "step":
        return (x > 0).to(x.dtype)
    raise ValueError(f"Unknown nonlinearity: {kind}")
