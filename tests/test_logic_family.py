from tensorlogic.logic import Domain, Relation, Rule, FixpointEngine


def test_ancestor_inference():
    people = Domain("Person", ["Alice", "Bob", "Charlie"])

    parent = Relation.from_pairs(
        "Parent",
        [people, people],
        [
            ("Alice", "Bob"),
            ("Bob", "Charlie"),
        ],
    )
    ancestor = Relation.empty_like(parent, "Ancestor")

    rules = [
        Rule("Ancestor", ["Parent"]),
        Rule("Ancestor", ["Parent", "Ancestor"]),
    ]

    engine = FixpointEngine(
        relations={"Parent": parent, "Ancestor": ancestor},
        rules=rules,
    )
    engine.run()

    pairs = set(ancestor.to_pairs())
    expected = {("Alice", "Bob"), ("Bob", "Charlie"), ("Alice", "Charlie")}
    assert pairs == expected
