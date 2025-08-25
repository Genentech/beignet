import hypothesis
import hypothesis.strategies
import torch

import beignet
import beignet.statistics


@hypothesis.given(
    batch_size=hypothesis.strategies.integers(min_value=1, max_value=5),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_two_one_sided_tests_one_sample_t_power(batch_size, dtype):
    d_vals = torch.tensor([0.0, 0.2, 0.4], dtype=dtype).repeat(batch_size, 1).flatten()
    n_vals = (
        torch.tensor([20.0, 50.0, 100.0], dtype=dtype).repeat(batch_size, 1).flatten()
    )
    low = torch.tensor(-0.2, dtype=dtype)
    high = torch.tensor(0.2, dtype=dtype)

    # Basic shape / dtype / range
    p = beignet.statistics.two_one_sided_tests_one_sample_t_power(
        d_vals, n_vals, low, high, alpha=0.05
    )
    assert p.shape == d_vals.shape
    assert p.dtype == dtype
    assert torch.all((p >= 0) & (p <= 1))

    # Monotonic in n for true effects within margins
    p_small = beignet.statistics.two_one_sided_tests_one_sample_t_power(
        d_vals, torch.full_like(d_vals, 20.0), low, high
    )
    p_large = beignet.statistics.two_one_sided_tests_one_sample_t_power(
        d_vals, torch.full_like(d_vals, 100.0), low, high
    )
    mask_within = (d_vals > low) & (d_vals < high)
    if torch.any(mask_within):
        assert torch.all(p_large[mask_within] >= p_small[mask_within])

    # Gradient and compile
    d_grad = d_vals.clone().requires_grad_(True)
    n_grad = n_vals.clone().requires_grad_(True)
    res = beignet.statistics.two_one_sided_tests_one_sample_t_power(
        d_grad, n_grad, low, high
    )
    res.sum().backward()
    assert d_grad.grad is not None
    assert n_grad.grad is not None

    compiled = torch.compile(
        beignet.statistics.two_one_sided_tests_one_sample_t_power, fullgraph=True
    )
    res_comp = compiled(d_vals, n_vals, low, high)
    assert torch.allclose(res, res_comp, atol=1e-5)

    # Cross-validate against SciPy noncentral t (loose tolerance due to approximations)
    import scipy.stats as stats

    for d0, n0 in [(0.0, 30), (0.1, 50), (0.15, 100)]:
        df = n0 - 1
        tcrit = stats.t.ppf(1 - 0.05, df)
        ncp_low = (d0 - float(low)) * (n0**0.5)
        ncp_high = (d0 - float(high)) * (n0**0.5)
        p1 = 1 - stats.nct.cdf(tcrit, df, ncp_low)
        p2 = stats.nct.cdf(-tcrit, df, ncp_high)
        ref = min(p1, p2)
        b = beignet.statistics.two_one_sided_tests_one_sample_t_power(
            torch.tensor(d0, dtype=dtype),
            torch.tensor(float(n0), dtype=dtype),
            low,
            high,
            alpha=0.05,
        )
        assert abs(float(b) - ref) < 0.2
