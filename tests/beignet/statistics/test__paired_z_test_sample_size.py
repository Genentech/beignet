import hypothesis
import hypothesis.strategies
import torch

import beignet
import beignet.statistics


@hypothesis.given(
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_paired_z_test_sample_size(dtype):
    d = torch.tensor(0.4, dtype=dtype)
    n = beignet.statistics.paired_z_test_sample_size(d, power=0.8, alpha=0.05)
    assert n.dtype == dtype
    assert n >= 1

    # Check achieved power
    p = beignet.statistics.paired_z_test_power(d, n, alpha=0.05)
    assert abs(float(p) - 0.8) < 0.15
