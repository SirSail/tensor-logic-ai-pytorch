# tensor-logic-ai-pytorch

> Practical implementation of Tensor Logic AI in PyTorch – Datalog reasoning as tensor operations (`einsum`) and neural networks in one unified framework

**Based on:** [Tensor Logic: The Language of AI](https://arxiv.org/abs/2510.12269) by Pedro Domingos

> "Progress in AI is hindered by the lack of a programming language with all the requisite features. [...] This paper proposes tensor logic, a language that solves these problems by unifying neural and symbolic AI at a fundamental level. The sole construct in tensor logic is the tensor equation, based on the observation that logical rules and Einstein summation are essentially the same operation."
> 
> — Pedro Domingos, *Tensor Logic: The Language of AI* (arXiv:2510.12269)

---

## Project goals

This project aims to:

- demonstrate understanding of **Tensor Logic AI** concepts in practice,
- show how to express logical reasoning (Datalog-style) and neural networks using the same tensor equation primitives,
- provide production-quality code: testable, modular, and readable.

The project is hybrid by design: it combines **logical reasoning** and **neural networks** in a single tensor-based model.

---

## Tensor Logic AI in a nutshell

Tensor Logic is based on the idea that:

- logical relations `R(x, y, ...)` can be represented as Boolean / [0, 1] tensors,
- Datalog rules can be written as tensor equations (e.g. `einsum` + threshold),
- reasoning (forward / backward chaining) is iterative execution of these equations,
- neural networks are also tensor programs (compositions of products and sums),
- the whole Tensor Logic program is differentiable, so its parameters can be trained with standard gradient-based methods.

This project implements a small, practical slice of that vision in PyTorch.

---

## Features

Current scope:

- **Core Tensor Logic program:**
  - definition of tensor variables and equations (`TensorVar`, `TensorEquation`),
  - execution of a tensor program via `torch.einsum` and simple nonlinearities,
  - proper PyTorch `nn.Module` integration for autograd and parameter management.

- **Logic layer (Datalog-like):**
  - relations as tensors (bool or [0, 1]) with named domains,
  - Datalog-style rules  
    `Head(x, z) :- Body1(x, y), Body2(y, z)`  
    defined via a Python API,
  - compilation of rules into tensor operations (`einsum` + threshold),
  - **forward-chaining** engine:
    - iterates rules until a fixpoint,
    - handles both base and derived relations.

- **Neural layer (NN as Tensor Logic):**
  - a simple MLP expressed as a `TensorProgram` (no `nn.Sequential` wrapper),
  - full PyTorch **autograd** support for training parameters.

- **Built-in demos:**
  - `family_tree` – infers `Ancestor` from `Parent` using Datalog-style rules,
  - `mlp_demo` – an MLP expressed as a tensor program with example forward pass.

---

## Installation

### Requirements

- Python >= 3.10  
- [PyTorch](https://pytorch.org/) (CPU is enough)  
- `pytest` (for tests)

### Setup

```bash
# Clone the repository
git clone https://github.com/SirSail/tensor-logic-ai-pytorch.git
cd tensor-logic-ai-pytorch

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

---

## Quick Start

### Example 1: Logic reasoning (Family tree)

```python
from tensorlogic.logic import Domain, Relation, Rule, FixpointEngine

# Define domain and base facts
people = Domain("Person", ["Alice", "Bob", "Charlie", "Diana"])
parent = Relation.from_pairs("Parent", [people, people], [
    ("Alice", "Bob"),
    ("Bob", "Charlie"),
    ("Alice", "Diana")
])

# Define derived relation and rules
ancestor = Relation.empty_like(parent, "Ancestor")
rules = [
    Rule("Ancestor", ["Parent"]),                    # Base case
    Rule("Ancestor", ["Parent", "Ancestor"])         # Recursive case
]

# Run inference
engine = FixpointEngine({"Parent": parent, "Ancestor": ancestor}, rules)
engine.run()

# View results
print("Inferred Ancestor relations:")
for a, b in sorted(ancestor.to_pairs()):
    print(f"  {a} -> {b}")
```

### Example 2: Neural network as Tensor Program

```python
import torch
from tensorlogic.nn import MLPConfig, MLPProgram

# Define MLP architecture
cfg = MLPConfig(input_dim=4, hidden_dims=[8, 4], output_dim=1)
model = MLPProgram(cfg)

# Forward pass
x = torch.randn(2, 4)
y = model(x)

print("Output shape:", y.shape)  # (2, 1)
```

---

## Running demos

```bash
# Logic demo
python -m tensorlogic.demos.family_tree

# Neural network demo
python -m tensorlogic.demos.mlp_demo
```

---

## Running tests

```bash
pytest tests/
```

---

## Project structure

```
tensor-logic-ai-pytorch/
├── src/tensorlogic/
│   ├── __init__.py          # Package metadata
│   ├── backend.py           # Low-level operations (einsum, nonlinearities)
│   ├── core.py              # TensorEquation, TensorProgram
│   ├── logic.py             # Domain, Relation, Rule, FixpointEngine
│   ├── nn.py                # MLPProgram (neural networks as tensor programs)
│   └── demos/
│       ├── family_tree.py   # Logic reasoning example
│       └── mlp_demo.py      # Neural network example
├── tests/
│   ├── test_core.py         # Core tensor program tests
│   ├── test_logic_family.py # Logic inference tests
│   ├── test_logic_fixpoint.py
│   └── test_nn_mlp.py       # Neural network tests
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Architecture highlights

### Name-based execution model

`TensorEquation` operates on **string names** rather than object references:

```python
TensorEquation(
    target="h0",
    sources=["x", "W0"],
    einsum_spec="bi,ij->bj",
    bias="b0",
    nonlinearity="relu"
)
```

This design allows:
- Clean separation between program structure and data
- Easy composition of complex computation graphs
- Stable references across multiple forward passes

### PyTorch integration

`TensorProgram` inherits from `nn.Module` with proper parameter management:

```python
class TensorProgram(nn.Module):
    def __init__(self, equations, parameters):
        super().__init__()
        self.equations = list(equations)
        self.params = nn.ParameterDict(parameters)
```

This ensures:
- Automatic parameter discovery for optimizers
- Full autograd support
- Seamless integration with PyTorch ecosystem

---

## Limitations (by design)

This is a minimal MVP focused on clarity and educational value:

- **Logic layer:** Currently supports only **binary relations** (arity = 2) for simplicity
- **Rules:** Limited to unary and binary rule patterns (transitive closure style)
- **Neural layer:** Basic MLP only (no convolutions, attention, etc.)

These limitations are intentional to keep the codebase readable and focused on core concepts.

---

## References

- Domingos, P. (2024). *Tensor Logic: The Language of AI*. arXiv:2510.12269. [[PDF]](https://arxiv.org/abs/2510.12269)

---

## License

MIT

---

## Author

SirSail - [GitHub](https://github.com/SirSail)

---

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/SirSail/tensor-logic-ai-pytorch/issues)