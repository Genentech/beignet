import hypothesis
import hypothesis.strategies as st
import torch

import beignet.statistics as S


@hypothesis.given(
    df1=st.integers(min_value=1, max_value=6),
    df2=st.integers(min_value=10, max_value=300),
    power=st.floats(min_value=0.6, max_value=0.95),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None, max_examples=20)
def test_f_test_mde(df1, df2, power, dtype):
    df1_t = torch.tensor(df1, dtype=dtype)
    df2_t = torch.tensor(df2, dtype=dtype)
    f2 = S.f_test_minimum_detectable_effect(df1_t, df2_t, power=power)
    assert f2.shape == df1_t.shape and f2.dtype == dtype
    p = S.f_test_power(f2, df1_t, df2_t)
    assert torch.all(p >= torch.tensor(power, dtype=dtype) - 0.1)

    f2_more = S.f_test_minimum_detectable_effect(df1_t, df2_t * 2, power=power)
    assert torch.all(f2_more <= f2 + 1e-6)

    out = torch.empty_like(df1_t)
    out_r = S.f_test_minimum_detectable_effect(df1_t, df2_t, power=power, out=out)
    assert torch.allclose(out, out_r)

    # Skip torch.compile(fullgraph=True) equivalence for CI stability
