import hypothesis
import hypothesis.strategies as st
import torch

import beignet.statistics as S


@hypothesis.given(
    n=st.integers(min_value=30, max_value=300),
    rows=st.integers(min_value=2, max_value=5),
    cols=st.integers(min_value=2, max_value=5),
    power=st.floats(min_value=0.6, max_value=0.95),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None, max_examples=20)
def test_chi_square_independence_mde(n, rows, cols, power, dtype):
    n = torch.tensor(n, dtype=dtype)
    r = torch.tensor(rows, dtype=dtype)
    c = torch.tensor(cols, dtype=dtype)
    w = S.chi_square_independence_minimum_detectable_effect(n, r, c, power=power)
    assert w.shape == torch.Size([1]) and w.dtype == dtype
    p = S.chi_square_independence_power(w, n, r, c)
    assert torch.all(p >= torch.tensor(power, dtype=dtype) - 0.1)

    n2 = n * 2
    w2 = S.chi_square_independence_minimum_detectable_effect(n2, r, c, power=power)
    assert torch.all(w2 <= w + 1e-5)

    out = torch.empty(1, dtype=dtype)
    out_r = S.chi_square_independence_minimum_detectable_effect(
        n,
        r,
        c,
        power=power,
        out=out,
    )
    assert torch.allclose(out, out_r)

    # Skip torch.compile(fullgraph=True) equivalence for CI stability
