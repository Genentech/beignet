import hypothesis
import hypothesis.strategies
import torch

import beignet
import beignet.statistics


@hypothesis.given(
    batch_size=hypothesis.strategies.integers(min_value=1, max_value=3),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_welch_t_test_sample_size(batch_size, dtype):
    d_vals = torch.tensor([0.3, 0.5], dtype=dtype).repeat(batch_size, 1).flatten()
    ratios = torch.tensor([1.0, 1.5], dtype=dtype).repeat(batch_size, 1).flatten()
    vr = torch.tensor(1.3, dtype=dtype)

    n1 = beignet.statistics.welch_t_test_sample_size(
        d_vals, ratio=ratios, var_ratio=vr, power=0.8, alpha=0.05
    )
    assert n1.shape == d_vals.shape
    assert n1.dtype == dtype
    assert torch.all(n1 >= 2)

    # Consistency: plug back to power
    n2 = torch.ceil(n1 * ratios)
    p = beignet.statistics.welch_t_test_power(d_vals, n1, n2, var_ratio=vr)
    # Allow some tolerance for iterative rounding
    assert torch.all(torch.abs(p - 0.8) < 0.1)

    # Cross-check a few points against statsmodels by scanning n1 nearby
    for d0 in [0.3, 0.5]:
        for r in [1.0, 1.2]:
            n1_est = beignet.statistics.welch_t_test_sample_size(
                torch.tensor(d0, dtype=dtype),
                ratio=torch.tensor(r, dtype=dtype),
                var_ratio=vr,
                power=0.8,
                alpha=0.05,
            )
            n2_est = torch.ceil(n1_est * r)
            p_est = beignet.statistics.welch_t_test_power(
                d0, n1_est, n2_est, var_ratio=vr
            )
            # Fallback to SciPy normal approx for reference
            import scipy.stats as stats

            n1i = int(n1_est.item())
            n2i = int(n2_est.item())
            se = ((1.0 / n1i) + (float(vr) / n2i)) ** 0.5
            ncp = float(d0) / se
            zcrit = stats.norm.ppf(1 - 0.05 / 2)
            ref = (1 - stats.norm.cdf(zcrit - ncp)) + stats.norm.cdf(-zcrit - ncp)
            assert abs(float(p_est) - ref) < 0.2
