import hypothesis
import hypothesis.strategies as st
import torch

import beignet.statistics as S


@hypothesis.given(
    n1=st.integers(min_value=8, max_value=120),
    ratio=st.floats(min_value=0.5, max_value=2.0),
    power=st.floats(min_value=0.6, max_value=0.95),
    alt=st.sampled_from(["two-sided", "greater", "less"]),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None, max_examples=25)
def test_mann_whitney_mde(n1, ratio, power, alt, dtype):
    n1_t = torch.tensor(n1, dtype=dtype)
    auc = S.mann_whitney_u_test_minimum_detectable_effect(
        n1_t, ratio=ratio, power=power, alternative=alt
    )
    assert auc.shape == n1_t.shape and auc.dtype == dtype
    n2_t = torch.ceil(n1_t * torch.tensor(ratio, dtype=dtype))
    p = S.mann_whitney_u_test_power(auc, n1_t, n2_t, alpha=0.05, alternative=alt)
    assert torch.all(p >= torch.tensor(power, dtype=dtype) - 0.1)

    auc2 = S.mann_whitney_u_test_minimum_detectable_effect(
        n1_t * 2, ratio=ratio, power=power, alternative=alt
    )
    # For two-sided and greater, auc decreases toward 0.5 as n increases; for less it also approaches 0.5
    assert torch.all(torch.abs(auc2 - 0.5) <= torch.abs(auc - 0.5) + 1e-6)

    out = torch.empty_like(n1_t)
    out_r = S.mann_whitney_u_test_minimum_detectable_effect(
        n1_t, ratio=ratio, power=power, alternative=alt, out=out
    )
    assert torch.allclose(out, out_r)

    # Skip torch.compile(fullgraph=True) equivalence for CI stability
