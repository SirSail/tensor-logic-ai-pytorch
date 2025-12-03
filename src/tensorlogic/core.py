from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping, Sequence
import torch
from torch import nn

from .backend import apply_einsum, apply_nonlinearity


@dataclass
class TensorEquation:
    """
    One tensor equation of the form:

        target = nonlinearity( einsum(einsum_spec, *sources) + bias )

    where sources, target and bias are names in the execution context.
    """
    target: str
    sources: Sequence[str]
    einsum_spec: str
    nonlinearity: str | None = None
    bias: str | None = None

    def execute(self, context: MutableMapping[str, torch.Tensor]) -> None:
        try:
            tensors = [context[name] for name in self.sources]
        except KeyError as e:
            missing = e.args[0]
            raise KeyError(f"Missing tensor '{missing}' for equation {self}") from e

        out = apply_einsum(self.einsum_spec, *tensors)
        if self.bias is not None:
            if self.bias not in context:
                raise KeyError(f"Missing bias tensor '{self.bias}' for equation {self}")
            out = out + context[self.bias]

        out = apply_nonlinearity(out, self.nonlinearity)
        context[self.target] = out


class TensorProgram(nn.Module):
    """
    A small differentiable program defined as a sequence of TensorEquations.

    Parameters are stored in a ParameterDict and live in the same name-space
    as tensors in the execution context.
    """

    def __init__(
        self,
        equations: Sequence[TensorEquation],
        parameters: Dict[str, nn.Parameter] | None = None,
    ) -> None:
        super().__init__()
        self.equations = list(equations)
        self.params = nn.ParameterDict(parameters or {})

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Executes the program with the given inputs.

        Inputs and parameters share a common namespace: if names collide,
        inputs override parameters (use unique names to avoid surprises).
        """
        context: Dict[str, torch.Tensor] = {}
        context.update(self.params)
        context.update(inputs)

        for eq in self.equations:
            eq.execute(context)

        return context
