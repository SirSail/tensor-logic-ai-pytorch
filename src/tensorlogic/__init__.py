"""
Tensor Logic AI – minimal differentiable reasoning engine in PyTorch.

Implements:
- Core tensor equations (`TensorVar`, `TensorEquation`, `TensorProgram`)
- Logic layer (relations, Datalog-style rules, forward-chaining)
- Neural layer (MLP as TensorProgram)

Author: Jakub Żegliński
License: MIT
"""
__version__ = "0.1.0"

from .core import TensorEquation, TensorProgram
from .nn import MLPProgram, MLPConfig
from .logic import Relation, Rule, Domain

__all__ = [
    "TensorEquation",
    "TensorProgram",
    "MLPProgram",
    "MLPConfig",
    "Relation",
    "Rule",
    "Domain",
]
