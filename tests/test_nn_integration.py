import pytest
import torch
from pydantic import ValidationError
from tensorlogic.nn import MLPConfig

def test_mlp_forward(small_mlp, random_batch):
    """Test standard forward pass with fixtures."""
    y = small_mlp(random_batch)
    assert y.shape == (4, 2)
    assert y.requires_grad

@pytest.mark.parametrize("batch_size", [1, 16, 128])
def test_mlp_broadcasting(small_mlp, batch_size):
    """Test MLP handles dynamic batch sizes correctly."""
    # input_dim is 10 from base_config fixture
    x = torch.randn(batch_size, 10)
    y = small_mlp(x)
    assert y.shape == (batch_size, 2)

def test_mlp_dimension_mismatch(small_mlp):
    """Test behavior when input dimension is wrong."""
    # Input dim is 10, passing 5 should fail during einsum or shape check
    x = torch.randn(4, 5)
    
    # Depending on how einsum/pytorch handles it, it might be RuntimeError
    # strict validation isn't in forward yet, but PyTorch will complain
    with pytest.raises(RuntimeError):
        small_mlp(x)

@pytest.mark.parametrize("config_kwargs", [
    {"input_dim": -1, "hidden_dims": [16], "output_dim": 5},
    {"input_dim": 10, "hidden_dims": [-5], "output_dim": 5},
    {"input_dim": 10, "hidden_dims": [], "output_dim": 5},
    {"input_dim": 10, "hidden_dims": [16], "output_dim": 0},
])
def test_invalid_config_raises_error(config_kwargs):
    """Verify Fail Fast behavior for invalid configuration using parametrization."""
    with pytest.raises(ValidationError):
        MLPConfig(**config_kwargs)
