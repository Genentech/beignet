import hypothesis
import hypothesis.strategies
import torch

import beignet
import beignet.statistics


@hypothesis.given(
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_mcnemars_test_power_and_sample_size(dtype):
    p01 = torch.tensor(0.2, dtype=dtype)
    p10 = torch.tensor(0.1, dtype=dtype)
    n = torch.tensor(200.0, dtype=dtype)

    pwr = beignet.statistics.mcnemars_test_power(
        p01, p10, n, alpha=0.05, two_sided=True
    )
    assert 0.0 <= pwr <= 1.0

    # Power increases with n
    small = beignet.statistics.mcnemars_test_power(
        p01, p10, torch.tensor(50.0, dtype=dtype)
    )
    large = beignet.statistics.mcnemars_test_power(
        p01, p10, torch.tensor(500.0, dtype=dtype)
    )
    assert large > small

    # Sample size consistency
    n_req = beignet.statistics.mcnemars_test_sample_size(
        p01, p10, power=0.8, alpha=0.05
    )
    p_ach = beignet.statistics.mcnemars_test_power(p01, p10, n_req, alpha=0.05)
    assert abs(float(p_ach) - 0.8) < 0.15
