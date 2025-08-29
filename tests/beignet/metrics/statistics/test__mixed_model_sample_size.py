import hypothesis.strategies
import pytest
import torch
from torch import Tensor

from beignet.metrics.statistics import MixedModelSampleSize


@hypothesis.given(
    effect_size=hypothesis.strategies.floats(min_value=0.1, max_value=1.0),
    cluster_size=hypothesis.strategies.integers(min_value=5, max_value=50),
    intraclass_correlation=hypothesis.strategies.floats(min_value=0.01, max_value=0.5),
    power=hypothesis.strategies.floats(min_value=0.7, max_value=0.95),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
def test_mixed_model_sample_size(
    effect_size,
    cluster_size,
    intraclass_correlation,
    power,
    alpha,
    dtype,
):
    metric = MixedModelSampleSize(power=power, alpha=alpha)

    effect_size_tensor = torch.tensor(effect_size, dtype=dtype)
    cluster_size_tensor = torch.tensor(cluster_size, dtype=dtype)
    icc_tensor = torch.tensor(intraclass_correlation, dtype=dtype)

    metric.update(effect_size_tensor, cluster_size_tensor, icc_tensor)
    result = metric.compute()

    assert isinstance(result, Tensor)
    assert result.item() > 0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
