import hypothesis
import hypothesis.strategies
import pytest
import torch
from torch import Tensor

from beignet.metrics.statistics import PartialEtaSquared


@hypothesis.given(
    sum_squares_effect=hypothesis.strategies.floats(min_value=1.0, max_value=100.0),
    sum_squares_error=hypothesis.strategies.floats(min_value=1.0, max_value=100.0),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
def test_partial_eta_squared(sum_squares_effect, sum_squares_error, dtype):
    metric = PartialEtaSquared()

    ss_effect_tensor = torch.tensor(sum_squares_effect, dtype=dtype)
    ss_error_tensor = torch.tensor(sum_squares_error, dtype=dtype)

    metric.update(ss_effect_tensor, ss_error_tensor)
    output = metric.compute()

    assert isinstance(output, Tensor)
    assert 0.0 <= output.item() <= 1.0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
