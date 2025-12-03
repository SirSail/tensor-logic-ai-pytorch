import torch

from tensorlogic.core import TensorEquation, TensorProgram


def test_simple_matrix_multiplication():
    a = torch.eye(2)
    b = torch.eye(2)
    eq = TensorEquation(target="c", sources=["a", "b"], einsum_spec="ij,jk->ik")
    prog = TensorProgram(equations=[eq], parameters={})

    out = prog({"a": a, "b": b})
    assert "c" in out
    assert torch.allclose(out["c"], torch.eye(2))
