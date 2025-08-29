import hypothesis
import hypothesis.strategies
import pytest
import torch
from torch import Tensor

from beignet.metrics.statistics import CohensKappaSampleSize


@hypothesis.given(
    kappa=hypothesis.strategies.floats(min_value=0.1, max_value=0.9),
    power=hypothesis.strategies.floats(min_value=0.7, max_value=0.95),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
def test_cohens_kappa_sample_size(
    kappa,
    power,
    alpha,
    dtype,
):
    metric = CohensKappaSampleSize(power=power, alpha=alpha)

    kappa_tensor = torch.tensor(kappa, dtype=dtype)

    metric.update(kappa_tensor)
    output = metric.compute()

    assert isinstance(output, Tensor)
    assert output.item() > 0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
