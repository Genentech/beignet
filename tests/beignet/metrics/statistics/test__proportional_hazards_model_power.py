import hypothesis.strategies
import pytest
import torch
from torch import Tensor

from beignet.metrics.statistics import ProportionalHazardsModelPower


@hypothesis.given(
    hazard_ratio=hypothesis.strategies.floats(min_value=1.1, max_value=5.0),
    sample_size=hypothesis.strategies.integers(min_value=10, max_value=500),
    event_probability=hypothesis.strategies.floats(min_value=0.1, max_value=0.9),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
def test_proportional_hazards_model_power(
    hazard_ratio,
    sample_size,
    event_probability,
    alpha,
    dtype,
):
    metric = ProportionalHazardsModelPower(alpha=alpha)

    hazard_ratio_tensor = torch.tensor(hazard_ratio, dtype=dtype)
    sample_size_tensor = torch.tensor(sample_size, dtype=dtype)
    event_probability_tensor = torch.tensor(event_probability, dtype=dtype)

    metric.update(hazard_ratio_tensor, sample_size_tensor, event_probability_tensor)
    result = metric.compute()

    assert isinstance(result, Tensor)
    assert 0.0 <= result.item() <= 1.0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
