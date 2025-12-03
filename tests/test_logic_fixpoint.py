from tensorlogic.logic import Domain, Relation, Rule, FixpointEngine


def test_fixpoint_stability():
    people = Domain("Person", ["A", "B"])

    parent = Relation.from_pairs(
        "Parent",
        [people, people],
        [("A", "B")],
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
    before = ancestor.tensor.clone()
    engine.run()
    after = ancestor.tensor.clone()

    assert (before == after).all()
