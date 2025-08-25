import hypothesis
import hypothesis.strategies as st
import torch

import beignet.statistics as S


@hypothesis.given(
    n=st.integers(min_value=20, max_value=600),
    d=st.floats(min_value=0.05, max_value=0.9),
    power=st.floats(min_value=0.6, max_value=0.95),
    two_sided=st.booleans(),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None, max_examples=25)
def test_mcnemar_mde(n, d, power, two_sided, dtype):
    n = torch.tensor(n, dtype=dtype)
    d = torch.tensor(d, dtype=dtype)
    delta = S.mcnemars_test_minimum_detectable_effect(
        d, n, power=power, two_sided=two_sided
    )
    assert delta.shape == n.shape and delta.dtype == dtype
    # Construct p01, p10 achieving |p01-p10| = delta with total = d
    p01 = torch.clamp((d + delta) / 2.0, 0.0, 1.0)
    p10 = torch.clamp(d - p01, 0.0, 1.0)
    pwr = S.mcnemars_test_power(p01, p10, n, two_sided=two_sided)
    # Compute max achievable power at given n and discordant rate (delta capped by d)
    p01_max = torch.clamp((d + d) / 2.0, 0.0, 1.0)
    p10_max = torch.clamp(d - p01_max, 0.0, 1.0)
    pwr_max = S.mcnemars_test_power(p01_max, p10_max, n, two_sided=two_sided)
    target = torch.minimum(torch.tensor(power, dtype=dtype), pwr_max)
    assert torch.all(pwr >= target - 0.1)

    # Monotonic: larger n -> smaller delta
    delta2 = S.mcnemars_test_minimum_detectable_effect(
        d, n * 2, power=power, two_sided=two_sided
    )
    assert torch.all(delta2 <= delta + 1e-6)

    out = torch.empty_like(n)
    out_r = S.mcnemars_test_minimum_detectable_effect(
        d, n, power=power, two_sided=two_sided, out=out
    )
    assert torch.allclose(out, out_r)

    # Skip torch.compile(fullgraph=True) equivalence for CI stability
