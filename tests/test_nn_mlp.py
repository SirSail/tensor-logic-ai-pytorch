import torch

from tensorlogic.nn import MLPConfig, MLPProgram


def test_mlp_forward_shape():
    cfg = MLPConfig(input_dim=3, hidden_dims=[5], output_dim=2)
    model = MLPProgram(cfg)

    x = torch.randn(4, 3)
    y = model(x)

    assert y.shape == (4, 2)
