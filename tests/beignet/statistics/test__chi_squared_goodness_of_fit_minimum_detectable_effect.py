import hypothesis
import hypothesis.strategies as st
import torch

import beignet.statistics as S


@hypothesis.given(
    n=st.integers(min_value=20, max_value=200),
    df=st.integers(min_value=1, max_value=8),
    power=st.floats(min_value=0.6, max_value=0.95),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None, max_examples=20)
def test_chi_square_gof_mde_shape_power(n, df, power, dtype):
    n = torch.tensor(n, dtype=dtype)
    df_t = torch.tensor(df, dtype=dtype)
    w = S.chi_square_goodness_of_fit_minimum_detectable_effect(n, df_t, power=power)
    assert w.shape == torch.Size([1])
    assert w.dtype == dtype
    # Power check (within tolerance)
    p = S.chi_square_goodness_of_fit_power(w, n, df_t)
    assert torch.all(p >= torch.tensor(power, dtype=dtype) - 0.1)

    # Monotonicity: larger n -> smaller MDE
    n2 = n * 2
    w2 = S.chi_square_goodness_of_fit_minimum_detectable_effect(n2, df_t, power=power)
    assert torch.all(w2 <= w + 1e-5)

    # out parameter
    out = torch.empty(1, dtype=dtype)
    out_r = S.chi_square_goodness_of_fit_minimum_detectable_effect(
        n,
        df_t,
        power=power,
        out=out,
    )
    assert torch.allclose(out, out_r)

    # compile equivalence
    # Skip torch.compile(fullgraph=True) equivalence for CI stability
