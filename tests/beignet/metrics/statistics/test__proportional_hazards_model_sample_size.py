import hypothesis.strategies
import pytest
import torch
from torch import Tensor

from beignet.metrics.statistics import ProportionalHazardsModelSampleSize


@hypothesis.given(
    hazard_ratio=hypothesis.strategies.floats(min_value=1.1, max_value=5.0),
    event_probability=hypothesis.strategies.floats(min_value=0.1, max_value=0.9),
    power=hypothesis.strategies.floats(min_value=0.7, max_value=0.95),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
def test_proportional_hazards_model_sample_size(
    hazard_ratio,
    event_probability,
    power,
    alpha,
    dtype,
):
    metric = ProportionalHazardsModelSampleSize(power=power, alpha=alpha)

    hazard_ratio_tensor = torch.tensor(hazard_ratio, dtype=dtype)
    event_probability_tensor = torch.tensor(event_probability, dtype=dtype)

    metric.update(hazard_ratio_tensor, event_probability_tensor)
    result = metric.compute()

    assert isinstance(result, Tensor)
    assert result.item() > 0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
