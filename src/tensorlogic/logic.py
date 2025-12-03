from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Sequence
import torch

from .backend import apply_einsum, apply_nonlinearity


@dataclass(frozen=True)
class Domain:
    """A finite domain of constants."""
    name: str
    elements: Sequence[str]


@dataclass
class Relation:
    """
    A logical relation represented as a Boolean tensor (stored as float in [0, 1]).
    For this minimal engine we focus on binary relations (arity = 2).
    """
    name: str
    domains: Sequence[Domain]
    tensor: torch.Tensor

    @classmethod
    def from_pairs(
        cls,
        name: str,
        domains: Sequence[Domain],
        pairs: Sequence[tuple],
    ) -> "Relation":
        shape = [len(d.elements) for d in domains]
        t = torch.zeros(*shape, dtype=torch.float32)
        index_maps = [
            {value: i for i, value in enumerate(d.elements)} for d in domains
        ]

        for pair in pairs:
            if len(pair) != len(domains):
                raise ValueError(
                    f"Pair {pair} has wrong arity for relation '{name}' "
                    f"(expected {len(domains)})"
                )
            idx = tuple(index_maps[i][v] for i, v in enumerate(pair))
            t[idx] = 1.0
        return cls(name, domains, t)

    @classmethod
    def empty_like(cls, other: "Relation", name: str) -> "Relation":
        return cls(name, other.domains, torch.zeros_like(other.tensor))

    def to_pairs(self) -> List[tuple]:
        coords = torch.nonzero(self.tensor > 0.5, as_tuple=False)
        result: List[tuple] = []
        for c in coords:
            tup = tuple(
                self.domains[i].elements[int(c[i])]
                for i in range(len(self.domains))
            )
            result.append(tup)
        return result


@dataclass
class Rule:
    """
    A minimal rule representation.

    For this MVP we support:
    - Unary rule:   Head(x, y) :- Body(x, y)
    - Binary rule:  Head(x, z) :- Body1(x, y), Body2(y, z)
    """
    head: str
    body: Sequence[str]


class FixpointEngine:
    """
    Forward-chaining inference engine implemented with tensor operations.

    Restriction (by design, for simplicity):
    - all relations are binary: R : D x D
    - rules are unary or binary as described in Rule docstring.
    """

    def __init__(self, relations: Dict[str, Relation], rules: Sequence[Rule]):
        self.relations = relations
        self.rules = list(rules)

    def run(self, max_iterations: int = 32) -> None:
        """Applies all rules until a fixpoint (or max_iterations)."""
        for _ in range(max_iterations):
            changed = False

            for rule in self.rules:
                head_rel = self.relations[rule.head]
                before = head_rel.tensor.clone()

                if len(rule.body) == 1:
                    # Head(x,y) :- Body(x,y)
                    body_rel = self.relations[rule.body[0]]
                    self._check_binary(head_rel, body_rel)
                    inferred = body_rel.tensor
                elif len(rule.body) == 2:
                    # Head(x,z) :- Body1(x,y), Body2(y,z)
                    r1 = self.relations[rule.body[0]]
                    r2 = self.relations[rule.body[1]]
                    self._check_binary(head_rel, r1, r2)
                    # Matrix-like composition over the shared middle dimension
                    inferred = apply_einsum("ij,jk->ik", r1.tensor, r2.tensor)
                    inferred = apply_nonlinearity(inferred, "step")
                else:
                    raise NotImplementedError(
                        f"Rules with {len(rule.body)} body literals are not supported."
                    )

                # OR in the Boolean semiring: 1 if either was 1
                head_rel.tensor = torch.clamp(head_rel.tensor + inferred, 0, 1)

                if not torch.equal(before, head_rel.tensor):
                    changed = True

            if not changed:
                break

    @staticmethod
    def _check_binary(*relations: Relation) -> None:
        for r in relations:
            if r.tensor.dim() != 2:
                raise ValueError(
                    f"Relation '{r.name}' must be binary (2D tensor), "
                    f"got shape {tuple(r.tensor.shape)}"
                )
