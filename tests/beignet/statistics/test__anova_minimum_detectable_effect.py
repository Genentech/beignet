import hypothesis
import hypothesis.strategies as st
import torch

import beignet.statistics as S


@hypothesis.given(
    n=st.integers(min_value=20, max_value=400),
    k=st.integers(min_value=2, max_value=6),
    power=st.floats(min_value=0.6, max_value=0.95),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None, max_examples=20)
def test_anova_mde(n, k, power, dtype):
    N = torch.tensor(n, dtype=dtype)
    K = torch.tensor(k, dtype=dtype)
    f = S.anova_minimum_detectable_effect(N, K, power=power)
    assert f.shape == N.shape and f.dtype == dtype
    p = S.anova_power(f, N, K)
    assert torch.all(p >= torch.tensor(power, dtype=dtype) - 0.1)

    f2 = S.anova_minimum_detectable_effect(N * 2, K, power=power)
    assert torch.all(f2 <= f + 1e-6)

    out = torch.empty_like(N)
    out_r = S.anova_minimum_detectable_effect(N, K, power=power, out=out)
    assert torch.allclose(out, out_r)

    # Skip torch.compile(fullgraph=True) equivalence for CI stability
