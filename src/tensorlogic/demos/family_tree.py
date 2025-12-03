import torch  # noqa: F401  # kept for potential extensions / debugging

from tensorlogic.logic import Domain, Relation, Rule, FixpointEngine


def main() -> None:
    people = Domain("Person", ["Alice", "Bob", "Charlie", "Diana"])

    parent = Relation.from_pairs(
        "Parent",
        [people, people],
        [
            ("Alice", "Bob"),
            ("Bob", "Charlie"),
            ("Alice", "Diana"),
        ],
    )

    ancestor = Relation.empty_like(parent, "Ancestor")

    rules = [
        # Ancestor(x, y) :- Parent(x, y).
        Rule("Ancestor", ["Parent"]),
        # Ancestor(x, z) :- Parent(x, y), Ancestor(y, z).
        Rule("Ancestor", ["Parent", "Ancestor"]),
    ]

    engine = FixpointEngine(
        relations={"Parent": parent, "Ancestor": ancestor},
        rules=rules,
    )
    engine.run()

    print("Inferred Ancestor relations:")
    for a, b in sorted(ancestor.to_pairs()):
        print(f"  {a} -> {b}")


if __name__ == "__main__":
    main()
