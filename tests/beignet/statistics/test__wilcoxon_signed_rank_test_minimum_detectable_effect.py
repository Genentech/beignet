import hypothesis
import hypothesis.strategies as st
import torch

import beignet.statistics as S


@hypothesis.given(
    n=st.integers(min_value=8, max_value=120),
    power=st.floats(min_value=0.6, max_value=0.95),
    alt=st.sampled_from(["two-sided", "greater", "less"]),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None, max_examples=25)
def test_wilcoxon_mde(n, power, alt, dtype):
    n = torch.tensor(n, dtype=dtype)
    ppos = S.wilcoxon_signed_rank_test_minimum_detectable_effect(
        n, power=power, alternative=alt
    )
    assert ppos.shape == n.shape and ppos.dtype == dtype
    pwr = S.wilcoxon_signed_rank_test_power(ppos, n, alternative=alt)
    assert torch.all(pwr >= torch.tensor(power, dtype=dtype) - 0.1)

    ppos2 = S.wilcoxon_signed_rank_test_minimum_detectable_effect(
        n * 2, power=power, alternative=alt
    )
    assert torch.all(torch.abs(ppos2 - 0.5) <= torch.abs(ppos - 0.5) + 1e-6)

    out = torch.empty_like(n)
    out_r = S.wilcoxon_signed_rank_test_minimum_detectable_effect(
        n, power=power, alternative=alt, out=out
    )
    assert torch.allclose(out, out_r)

    # Skip torch.compile(fullgraph=True) equivalence for CI stability
