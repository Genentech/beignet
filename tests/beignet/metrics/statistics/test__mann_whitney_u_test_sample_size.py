import hypothesis
import hypothesis.strategies
import pytest
import torch
from torch import Tensor

from beignet.metrics.statistics import MannWhitneyUTestSampleSize


@hypothesis.given(
    auc=hypothesis.strategies.floats(min_value=0.5, max_value=1.0),
    sample_ratio=hypothesis.strategies.floats(min_value=0.5, max_value=2.0),
    power=hypothesis.strategies.floats(min_value=0.7, max_value=0.95),
    alpha=hypothesis.strategies.floats(min_value=0.01, max_value=0.1),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
def test_mann_whitney_u_test_sample_size(
    auc,
    sample_ratio,
    power,
    alpha,
    dtype,
):
    metric = MannWhitneyUTestSampleSize(power=power, alpha=alpha)

    auc_tensor = torch.tensor(auc, dtype=dtype)
    sample_ratio_tensor = torch.tensor(sample_ratio, dtype=dtype)

    metric.update(auc_tensor, sample_ratio_tensor)
    output = metric.compute()

    assert isinstance(output, Tensor)
    assert output.item() > 0

    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()
