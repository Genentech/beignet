import hypothesis
import hypothesis.strategies as st
import torch

import beignet.statistics as stats


@hypothesis.given(dtype=st.sampled_from([torch.float32, torch.float64]))
@hypothesis.settings(deadline=None)
def test_ancova_power_and_sample_size(dtype):
    f = torch.tensor(0.25, dtype=dtype)
    k = torch.tensor(3.0, dtype=dtype)
    r2 = torch.tensor(0.4, dtype=dtype)
    p = torch.tensor(1.0, dtype=dtype)

    N = torch.tensor(120.0, dtype=dtype)
    pw = stats.analysis_of_covariance_power(f, N, k, r2, p, alpha=0.05)
    assert 0.0 <= pw <= 1.0

    small = stats.analysis_of_covariance_power(
        f,
        torch.tensor(80.0, dtype=dtype),
        k,
        r2,
        p,
    )
    large = stats.analysis_of_covariance_power(
        f,
        torch.tensor(200.0, dtype=dtype),
        k,
        r2,
        p,
    )
    assert large > small

    N_req = stats.analysis_of_covariance_sample_size(f, k, r2, p, power=0.8, alpha=0.05)
    p_ach = stats.analysis_of_covariance_power(f, N_req, k, r2, p, alpha=0.05)
    assert abs(float(p_ach) - 0.8) < 0.2
