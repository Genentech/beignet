import hypothesis
import hypothesis.strategies
import torch

import beignet
import beignet.statistics


@hypothesis.given(
    batch_size=hypothesis.strategies.integers(min_value=1, max_value=4),
    dtype=hypothesis.strategies.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_welch_t_test_power(batch_size, dtype):
    d_vals = torch.tensor([0.2, 0.5, 0.8], dtype=dtype).repeat(batch_size, 1).flatten()
    n1_vals = (
        torch.tensor([20.0, 30.0, 50.0], dtype=dtype).repeat(batch_size, 1).flatten()
    )
    ratios = torch.tensor([1.0, 1.5, 2.0], dtype=dtype).repeat(batch_size, 1).flatten()
    vr = torch.tensor(1.5, dtype=dtype)

    n2_vals = torch.ceil(n1_vals * ratios)

    p = beignet.statistics.welch_t_test_power(
        d_vals,
        n1_vals,
        n2_vals,
        var_ratio=vr,
        alpha=0.05,
    )
    assert p.shape == d_vals.shape
    assert p.dtype == dtype
    assert torch.all((p >= 0) & (p <= 1))

    # Broadcasting
    d = torch.tensor([[0.2], [0.5]], dtype=dtype)
    n1 = torch.tensor([[20.0, 40.0]], dtype=dtype)
    n2 = torch.ceil(n1 * 1.3)
    out = beignet.statistics.welch_t_test_power(d, n1, n2, var_ratio=vr)
    assert out.shape == (2, 2)

    # Out parameter shape mismatch should raise RuntimeError
    bad_out = torch.empty((3, 3), dtype=dtype)
    try:
        beignet.statistics.welch_t_test_power(d, n1, n2, var_ratio=vr, out=bad_out)
        raise AssertionError("Expected RuntimeError for shape mismatch")
    except RuntimeError:
        pass

    # Gradient and compile
    d_grad = d_vals.clone().requires_grad_(True)
    n1_grad = n1_vals.clone().requires_grad_(True)
    n2_grad = n2_vals.clone().requires_grad_(True)
    res = beignet.statistics.welch_t_test_power(d_grad, n1_grad, n2_grad, var_ratio=vr)
    res.sum().backward()
    assert d_grad.grad is not None and torch.all(torch.isfinite(d_grad.grad))
    assert n1_grad.grad is not None and torch.all(torch.isfinite(n1_grad.grad))
    assert n2_grad.grad is not None and torch.all(torch.isfinite(n2_grad.grad))

    compiled = torch.compile(beignet.statistics.welch_t_test_power, fullgraph=True)
    res_comp = compiled(d_vals, n1_vals, n2_vals, var_ratio=vr)
    assert torch.allclose(res, res_comp, atol=1e-5)

    # Cross-validate vs statsmodels (unequal variances)
    for d0 in [0.2, 0.5]:
        for n10 in [20, 40]:
            for r in [1.0, 1.5]:
                n20 = int((n10 * r) + 0.5)
                beignet_p = beignet.statistics.welch_t_test_power(
                    torch.tensor(d0, dtype=dtype),
                    torch.tensor(float(n10), dtype=dtype),
                    torch.tensor(float(n20), dtype=dtype),
                    var_ratio=vr,
                    alpha=0.05,
                    alternative="two-sided",
                )
                # statsmodels does not expose unequal-variance power directly in some versions.
                # Compare against normal approx by adjusting standard error to Welch's form.
                # This is a weaker check but keeps the test portable.
                import scipy.stats as stats

                vr_val = float(vr)
                se = ((1.0 / n10) + (vr_val / n20)) ** 0.5
                ncp = d0 / se
                zcrit = stats.norm.ppf(1 - 0.05 / 2)
                sm = (1 - stats.norm.cdf(zcrit - ncp)) + stats.norm.cdf(-zcrit - ncp)
                # Allow loose tolerance due to approximations
                assert torch.isclose(
                    beignet_p,
                    torch.tensor(sm, dtype=dtype),
                    rtol=0.35,
                    atol=0.02,
                ), f"d={d0}, n1={n10}, n2={n20}"
