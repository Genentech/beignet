
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from beignet.metrics.statistics import ProportionalHazardsModelSampleSize


@given(
    hazard_ratio=st.floats(min_value=1.1, max_value=5.0),
    event_probability=st.floats(min_value=0.1, max_value=0.9),
    power=st.floats(min_value=0.7, max_value=0.95),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64]),
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

    assert isinstance(result, torch.Tensor)
    assert result.item() > 0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
