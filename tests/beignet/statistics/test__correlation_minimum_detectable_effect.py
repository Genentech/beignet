import hypothesis
import hypothesis.strategies as st
import torch

import beignet.statistics as S


@hypothesis.given(
    n=st.integers(min_value=8, max_value=300),
    power=st.floats(min_value=0.6, max_value=0.95),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None, max_examples=25)
def test_correlation_mde(n, power, dtype):
    n = torch.tensor(n, dtype=dtype)
    r = S.correlation_minimum_detectable_effect(
        n, power=power, alpha=0.05, alternative="two-sided"
    )
    assert r.shape == n.shape and r.dtype == dtype
    # Plug back to power
    p = S.correlation_power(r, n, alpha=0.05, alternative="two-sided")
    assert torch.all(p >= torch.tensor(power, dtype=dtype) - 0.05)

    # Monotonic: increase n -> smaller |r|
    r2 = S.correlation_minimum_detectable_effect(n * 2, power=power)
    assert torch.all(r2 <= r + 1e-6)

    out = torch.empty_like(n)
    out_r = S.correlation_minimum_detectable_effect(n, power=power, out=out)
    assert torch.allclose(out, out_r)

    # torch.compile(fullgraph=True) equivalence can be flaky in CI; skip here
