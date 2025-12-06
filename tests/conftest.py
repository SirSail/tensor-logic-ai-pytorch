import pytest
import torch
from tensorlogic.nn import MLPConfig, MLPProgram

@pytest.fixture
def base_config():
    return MLPConfig(
        input_dim=10,
        hidden_dims=[16, 8],
        output_dim=2
    )

@pytest.fixture
def small_mlp(base_config):
    return MLPProgram(base_config)

@pytest.fixture
def random_batch():
    torch.manual_seed(42)
    # [batch_size=4, input_dim=10] - matching base_config input_dim
    return torch.randn(4, 10)
