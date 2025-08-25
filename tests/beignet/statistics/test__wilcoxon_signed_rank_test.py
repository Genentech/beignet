import hypothesis
import hypothesis.strategies as st
import torch

import beignet.statistics as stats


@hypothesis.given(dtype=st.sampled_from([torch.float32, torch.float64]))
@hypothesis.settings(deadline=None)
def test_wilcoxon_signed_rank_power_and_sample_size(dtype):
    p = torch.tensor(0.65, dtype=dtype)
    n = torch.tensor(40.0, dtype=dtype)

    pw = stats.wilcoxon_signed_rank_test_power(p, n, alpha=0.05)
    assert 0.0 <= pw <= 1.0

    small = stats.wilcoxon_signed_rank_test_power(p, torch.tensor(20.0, dtype=dtype))
    large = stats.wilcoxon_signed_rank_test_power(p, torch.tensor(100.0, dtype=dtype))
    assert large > small

    n_req = stats.wilcoxon_signed_rank_test_sample_size(p, power=0.8, alpha=0.05)
    p_ach = stats.wilcoxon_signed_rank_test_power(p, n_req, alpha=0.05)
    assert abs(float(p_ach) - 0.8) < 0.2
