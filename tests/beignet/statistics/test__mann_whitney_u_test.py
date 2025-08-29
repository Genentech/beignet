import hypothesis
import hypothesis.strategies
import torch

import beignet
import beignet.statistics


@hypothesis.given(
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_mann_whitney_u_test_power_and_sample_size(dtype):
    auc = torch.tensor(0.65, dtype=dtype)
    n1 = torch.tensor(40.0, dtype=dtype)
    r = torch.tensor(1.0, dtype=dtype)

    pwr = beignet.statistics.mann_whitney_u_test_power(auc, n1, ratio=r, alpha=0.05)
    assert 0.0 <= pwr <= 1.0

    # Power increases with n1
    small = beignet.statistics.mann_whitney_u_test_power(
        auc,
        torch.tensor(20.0, dtype=dtype),
        ratio=r,
    )
    large = beignet.statistics.mann_whitney_u_test_power(
        auc,
        torch.tensor(100.0, dtype=dtype),
        ratio=r,
    )
    assert large > small

    # Sample size consistency (rough)
    n_req = beignet.statistics.mann_whitney_u_test_sample_size(
        auc,
        ratio=r,
        power=0.8,
        alpha=0.05,
    )
    p_ach = beignet.statistics.mann_whitney_u_test_power(
        auc,
        n_req,
        ratio=r,
        alpha=0.05,
    )
    assert abs(float(p_ach) - 0.8) < 0.2
