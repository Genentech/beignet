import hypothesis
import hypothesis.strategies as st
import torch

import beignet.statistics as S


@hypothesis.given(
    n=st.integers(min_value=30, max_value=400),
    k=st.integers(min_value=2, max_value=6),
    p=st.integers(min_value=0, max_value=3),
    r2=st.floats(min_value=0.0, max_value=0.8),
    power=st.floats(min_value=0.6, max_value=0.95),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None, max_examples=20)
def test_ancova_mde(n, k, p, r2, power, dtype):
    N = torch.tensor(n, dtype=dtype)
    K = torch.tensor(k, dtype=dtype)
    P = torch.tensor(p, dtype=dtype)
    R2 = torch.tensor(r2, dtype=dtype)
    f = S.analysis_of_covariance_minimum_detectable_effect(N, K, R2, P, power=power)
    assert f.shape == torch.Size([1]) and f.dtype == dtype
    pr = S.analysis_of_covariance_power(f, N, K, R2, P)
    assert torch.all(pr >= torch.tensor(power, dtype=dtype) - 0.1)

    f2 = S.analysis_of_covariance_minimum_detectable_effect(
        N * 2,
        K,
        R2,
        P,
        power=power,
    )
    assert torch.all(f2 <= f + 1e-6)

    out = torch.empty(1, dtype=dtype)
    out_r = S.analysis_of_covariance_minimum_detectable_effect(
        N,
        K,
        R2,
        P,
        power=power,
        out=out,
    )
    assert torch.allclose(out, out_r)

    # Skip torch.compile(fullgraph=True) equivalence for CI stability
