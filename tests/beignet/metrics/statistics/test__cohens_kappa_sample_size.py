"""Test CohensKappaSampleSize metric."""

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from beignet.metrics.statistics import CohensKappaSampleSize


@given(
    kappa=st.floats(min_value=0.1, max_value=0.9),
    categories=st.integers(min_value=2, max_value=5),
    power=st.floats(min_value=0.7, max_value=0.95),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
def test_cohens_kappa_sample_size(
    kappa,
    categories,
    power,
    alpha,
    dtype,
):
    metric = CohensKappaSampleSize(power=power, alpha=alpha)

    kappa_tensor = torch.tensor(kappa, dtype=dtype)
    categories_tensor = torch.tensor(categories, dtype=dtype)

    metric.update(kappa_tensor, categories_tensor)
    result = metric.compute()

    assert isinstance(result, torch.Tensor)
    assert result.item() > 0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
