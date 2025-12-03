# tensor-logic-ai-pytorch

> Minimal Tensor Logic AI engine in PyTorch – Datalog reasoning as tensor operations (`einsum`) and neural networks in one framework

A minimal Tensor Logic AI engine in Python + PyTorch. This project shows how to:

- express logical rules (Datalog-style) as tensor equations (`einsum`),
- represent classical neural networks (e.g. MLPs) as the same kind of tensor program,
- use PyTorch autograd to train parameters of the whole Tensor Logic program.

---

## Project goals

This project aims to:

- demonstrate understanding of **Tensor Logic AI** (Pedro Domingos),
- show how to apply it in practice on small, transparent examples,
- provide production-quality code: testable, modular, and readable.

The project is hybrid by design: it combines **logical reasoning** and **neural networks** in a single tensor-based model.

---

## Tensor Logic AI in a nutshell

Tensor Logic is based on the idea that:

- logical relations `R(x, y, ...)` can be represented as Boolean / [0, 1] tensors,
- Datalog rules can be written as tensor equations (e.g. `einsum` + threshold),
- reasoning (forward / backward chaining) is iterative execution of these equations,
- neural networks are also tensor programs (compositions of products and sums),
- the whole Tensor Logic program is differentiable, so its parameters can be trained
  with standard gradient-based methods.

This project implements a small, practical slice of that vision in PyTorch.

---

## Features

Current scope:

- **Core Tensor Logic program:**
  - definition of tensor variables and equations (`TensorVar`, `TensorEquation`),
  - execution of a tensor program via `torch.einsum` and simple nonlinearities.

- **Logic layer (Datalog-like):**
  - relations as tensors (bool or [0, 1]) with named domains,
  - Datalog-style rules  
    `Head(x, z) :- Body1(x, y), Body2(y, z), ...`  
    defined via a Python API,
  - compilation of rules into tensor operations (`einsum` + threshold),
  - **forward-chaining** engine:
    - iterates rules until a fixpoint,
    - handles both base and derived relations.

- **Neural layer (NN as Tensor Logic):**
  - a simple MLP expressed as a `TensorProgram` (no `nn.Sequential` wrapper),
  - ability to use PyTorch **autograd** to train MLP parameters.

- **Built-in demos:**
  - `family_tree` – infers `Ancestor` from `Parent` using Datalog-style rules,
  - `mlp_demo` – an MLP expressed as a tensor program, with an example run and (optionally) a small training loop.

---

## Requirements

- Python >= 3.10  
- [PyTorch](https://pytorch.org/) (CPU is enough)  
- `numpy`  
- `pytest` (for tests)



```bash
pip install -r requirements.txt
