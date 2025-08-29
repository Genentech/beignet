
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from beignet.metrics.statistics import CohensKappaPower


@given(
    kappa=st.floats(min_value=0.1, max_value=0.9),
    sample_size=st.integers(min_value=20, max_value=200),
    categories=st.integers(min_value=2, max_value=5),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
def test_cohens_kappa_power(
    kappa,
    sample_size,
    categories,
    alpha,
    dtype,
):
    metric = CohensKappaPower(alpha=alpha)

    kappa_tensor = torch.tensor(kappa, dtype=dtype)
    sample_size_tensor = torch.tensor(sample_size, dtype=dtype)
    categories_tensor = torch.tensor(categories, dtype=dtype)

    metric.update(kappa_tensor, sample_size_tensor, categories_tensor)
    result = metric.compute()

    assert isinstance(result, torch.Tensor)
    assert 0.0 <= result.item() <= 1.0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
