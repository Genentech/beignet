import hypothesis
import hypothesis.strategies
import torch

import beignet
import beignet.statistics


@hypothesis.given(
    batch_size=hypothesis.strategies.integers(min_value=1, max_value=8),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_paired_z_test_power(batch_size, dtype):
    d_vals = torch.tensor([0.0, 0.2, 0.5], dtype=dtype).repeat(batch_size, 1).flatten()
    n_vals = (
        torch.tensor([10.0, 30.0, 100.0], dtype=dtype).repeat(batch_size, 1).flatten()
    )

    p = beignet.statistics.paired_z_test_power(d_vals, n_vals, alpha=0.05)
    assert p.shape == d_vals.shape
    assert p.dtype == dtype
    assert torch.all((p >= 0) & (p <= 1))

    # Monotonic in n
    p_small = beignet.statistics.paired_z_test_power(
        d_vals,
        torch.full_like(d_vals, 10.0),
    )
    p_large = beignet.statistics.paired_z_test_power(
        d_vals,
        torch.full_like(d_vals, 100.0),
    )
    assert torch.all(p_large >= p_small)

    # Cross-validate exactly with Normal CDF
    import scipy.stats as stats

    for d0, n0, alt in [
        (0.2, 30, "two-sided"),
        (0.2, 30, "greater"),
        (0.2, 30, "less"),
    ]:
        b = beignet.statistics.paired_z_test_power(
            torch.tensor(d0, dtype=dtype),
            torch.tensor(float(n0), dtype=dtype),
            alpha=0.05,
            alternative=alt,
        )
        ncp = d0 * (n0**0.5)
        if alt == "two-sided":
            z = stats.norm.ppf(1 - 0.05 / 2)
            ref = (1 - stats.norm.cdf(z - ncp)) + stats.norm.cdf(-z - ncp)
        elif alt == "greater":
            z = stats.norm.ppf(1 - 0.05)
            ref = 1 - stats.norm.cdf(z - ncp)
        else:
            z = stats.norm.ppf(1 - 0.05)
            ref = stats.norm.cdf(-z - ncp)
        assert abs(float(b) - ref) < 0.08
