import hypothesis
import hypothesis.strategies as st
import torch

import beignet.metrics.functional.statistics
import beignet.statistics


@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=10),
    effect_size=st.floats(min_value=0.1, max_value=0.8),
    power=st.floats(min_value=0.1, max_value=0.95),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_proportion_sample_size(batch_size, effect_size, power, alpha, dtype):
    """Test proportion_sample_size functional wrapper."""
    effect_size_tensor = torch.full((batch_size,), effect_size, dtype=dtype)
    power_tensor = torch.full((batch_size,), power, dtype=dtype)

    result_functional = beignet.metrics.functional.statistics.proportion_sample_size(
        effect_size_tensor,
        power_tensor,
        alpha=alpha,
    )
    result_direct = beignet.statistics.proportion_sample_size(
        effect_size_tensor,
        power_tensor,
        alpha=alpha,
    )

    assert torch.allclose(result_functional, result_direct, atol=1e-6)
    assert torch.all(result_functional >= 1.0)
