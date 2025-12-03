from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import torch
from torch import nn

from .core import TensorEquation, TensorProgram


@dataclass
class MLPConfig:
    input_dim: int
    hidden_dims: List[int]
    output_dim: int


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
        prev_dim = cfg.input_dim

        # Hidden layers
        for layer_idx, hidden_dim in enumerate(cfg.hidden_dims):
            W_name = f"W{layer_idx}"
            b_name = f"b{layer_idx}"
            out_name = f"h{layer_idx}"

            params[W_name] = nn.Parameter(
                torch.randn(prev_dim, hidden_dim) * 0.1
            )
            params[b_name] = nn.Parameter(torch.zeros(hidden_dim))

            equations.append(
                TensorEquation(
                    target=out_name,
                    sources=[input_name, W_name],
                    einsum_spec="bi,ij->bj",
                    nonlinearity="relu",
                    bias=b_name,
                )
            )

            input_name = out_name
            prev_dim = hidden_dim

        # Output layer
        W_name = "W_out"
        b_name = "b_out"
        output_name = "y"

        params[W_name] = nn.Parameter(
            torch.randn(prev_dim, cfg.output_dim) * 0.1
        )
        params[b_name] = nn.Parameter(torch.zeros(cfg.output_dim))

        equations.append(
            TensorEquation(
                target=output_name,
                sources=[input_name, W_name],
                einsum_spec="bi,ij->bj",
                nonlinearity=None,
                bias=b_name,
            )
        )

        super().__init__(equations=equations, parameters=params)
        self.input_name = "x"
        self.output_name = output_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the MLP on input x of shape [batch, input_dim] and returns y.
        """
        context = super().forward({self.input_name: x})
        return context[self.output_name]
