import hypothesis
import hypothesis.strategies
import pytest
import torch
from torch import Tensor

from beignet.metrics.statistics import CramersV


@hypothesis.given(
    chi_square=hypothesis.strategies.floats(min_value=0.1, max_value=10.0),
    sample_size=hypothesis.strategies.integers(min_value=10, max_value=100),
    min_dim=hypothesis.strategies.integers(min_value=2, max_value=5),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
def test_cramers_v(chi_square, sample_size, min_dim, dtype):
    metric = CramersV()

    chi_square_tensor = torch.tensor(chi_square, dtype=dtype)
    sample_size_tensor = torch.tensor(sample_size, dtype=dtype)
    min_dim_tensor = torch.tensor(min_dim, dtype=dtype)

    metric.update(chi_square_tensor, sample_size_tensor, min_dim_tensor)
    output = metric.compute()

    assert isinstance(output, Tensor)
    assert 0.0 <= output.item() <= 1.0  # Cramer's V should be between 0 and 1

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
