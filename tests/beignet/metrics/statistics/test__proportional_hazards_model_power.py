
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from beignet.metrics.statistics import ProportionalHazardsModelPower


@given(
    hazard_ratio=st.floats(min_value=1.1, max_value=5.0),
    sample_size=st.integers(min_value=10, max_value=500),
    event_probability=st.floats(min_value=0.1, max_value=0.9),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64]),
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

    assert isinstance(result, torch.Tensor)
    assert 0.0 <= result.item() <= 1.0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
