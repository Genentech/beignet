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
def test_two_one_sided_tests_two_sample_t_power(batch_size, dtype):
    d_vals = torch.tensor([0.0, 0.2, 0.4], dtype=dtype).repeat(batch_size, 1).flatten()
    n1_vals = (
        torch.tensor([20.0, 40.0, 80.0], dtype=dtype).repeat(batch_size, 1).flatten()
    )
    ratio = torch.tensor(1.5, dtype=dtype)
    low = torch.tensor(-0.2, dtype=dtype)
    high = torch.tensor(0.2, dtype=dtype)

    # Basic shape / dtype / range
    p = beignet.statistics.two_one_sided_tests_two_sample_t_power(
        d_vals,
        n1_vals,
        ratio=ratio,
        low=low,
        high=high,
        alpha=0.05,
    )
    assert p.shape == d_vals.shape
    assert p.dtype == dtype
    assert torch.all((p >= 0) & (p <= 1))

    # Monotonic in n1 for true effects within margins
    p_small = beignet.statistics.two_one_sided_tests_two_sample_t_power(
        d_vals,
        torch.full_like(d_vals, 20.0),
        ratio=ratio,
        low=low,
        high=high,
    )
    p_large = beignet.statistics.two_one_sided_tests_two_sample_t_power(
        d_vals,
        torch.full_like(d_vals, 100.0),
        ratio=ratio,
        low=low,
        high=high,
    )
    mask_within = (d_vals > low) & (d_vals < high)
    if torch.any(mask_within):
        assert torch.all(p_large[mask_within] >= p_small[mask_within])

    # Gradient and compile
    d_grad = d_vals.clone().requires_grad_(True)
    n1_grad = n1_vals.clone().requires_grad_(True)
    res = beignet.statistics.two_one_sided_tests_two_sample_t_power(
        d_grad,
        n1_grad,
        ratio=ratio,
        low=low,
        high=high,
    )
    res.sum().backward()
    assert d_grad.grad is not None
    assert n1_grad.grad is not None

    compiled = torch.compile(
        beignet.statistics.two_one_sided_tests_two_sample_t_power,
        fullgraph=True,
    )
    res_comp = compiled(d_vals, n1_vals, ratio=ratio, low=low, high=high)
    assert torch.allclose(res, res_comp, atol=1e-5)

    # Cross-validate against SciPy nct (loose tolerance)
    import scipy.stats as stats

    for d0, n10, r in [(0.0, 30, 1.0), (0.1, 40, 1.5)]:
        n20 = int(n10 * r)
        df = n10 + n20 - 2
        tcrit = stats.t.ppf(1 - 0.05, df)
        se = (1 / n10 + 1 / n20) ** 0.5
        ncp_low = (d0 - float(low)) / se
        ncp_high = (d0 - float(high)) / se
        p1 = 1 - stats.nct.cdf(tcrit, df, ncp_low)
        p2 = stats.nct.cdf(-tcrit, df, ncp_high)
        ref = min(p1, p2)
        b = beignet.statistics.two_one_sided_tests_two_sample_t_power(
            torch.tensor(d0, dtype=dtype),
            torch.tensor(float(n10), dtype=dtype),
            ratio=torch.tensor(float(r), dtype=dtype),
            low=low,
            high=high,
        )
        assert abs(float(b) - ref) < 0.2
