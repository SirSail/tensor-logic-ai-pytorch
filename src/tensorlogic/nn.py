from __future__ import annotations
from pydantic import BaseModel, field_validator
from typing import Dict, List
import torch
from torch import nn

from .core import TensorEquation, TensorProgram


class MLPConfig(BaseModel):
    input_dim: int
    hidden_dims: List[int]
    output_dim: int

    @field_validator("input_dim", "output_dim")
    @classmethod
    def check_positive(cls, v: int, info) -> int:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return v

    @field_validator("hidden_dims")
    @classmethod
    def check_hidden_dims(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("Hidden dimensions list cannot be empty")
        if any(d <= 0 for d in v):
            raise ValueError("All hidden dimensions must be positive")
        return v


class MLPProgram(TensorProgram):
    """
    A simple fully-connected MLP expressed as a TensorProgram.

    Public API:
        y = model(x)   # x: [batch, input_dim], y: [batch, output_dim]
    """

    def __init__(self, cfg: MLPConfig):
        equations: List[TensorEquation] = []
        params: Dict[str, nn.Parameter] = {}

        input_name = "x"  # public input
    def __init__(self, cfg: MLPConfig):
        equations: List[TensorEquation] = []
        params: Dict[str, nn.Parameter] = {}

        input_name = "x"  # public input
        prev_dim = cfg.input_dim

        # Hidden layers
        for layer_idx, hidden_dim in enumerate(cfg.hidden_dims):
            input_name, prev_dim = self._add_layer(
                params=params,
                equations=equations,
                input_name=input_name,
                prev_dim=prev_dim,
                out_dim=hidden_dim,
                layer_suffix=str(layer_idx),
                nonlinearity="relu",
            )

        # Output layer
        output_name, _ = self._add_layer(
            params=params,
            equations=equations,
            input_name=input_name,
            prev_dim=prev_dim,
            out_dim=cfg.output_dim,
            layer_suffix="out",
            nonlinearity=None,
            target_name="y",
        )

        super().__init__(equations=equations, parameters=params)
        self.input_name = "x"
        self.output_name = output_name

    def _add_layer(
        self,
        params: Dict[str, nn.Parameter],
        equations: List[TensorEquation],
        input_name: str,
        prev_dim: int,
        out_dim: int,
        layer_suffix: str,
        nonlinearity: str | None,
        target_name: str | None = None,
    ) -> tuple[str, int]:
        """Helper to create a linear layer with optional nonlinearity."""
        W_name = f"W{layer_suffix}"
        b_name = f"b{layer_suffix}"
        out_name = target_name if target_name else f"h{layer_suffix}"

        params[W_name] = nn.Parameter(torch.randn(prev_dim, out_dim) * 0.1)
        params[b_name] = nn.Parameter(torch.zeros(out_dim))

        equations.append(
            TensorEquation(
                target=out_name,
                sources=[input_name, W_name],
                einsum_spec="bi,ij->bj",
                nonlinearity=nonlinearity,
                bias=b_name,
            )
        )
        return out_name, out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the MLP on input x of shape [batch, input_dim] and returns y.
        """
        context = super().forward({self.input_name: x})
        return context[self.output_name]
