import hypothesis
import hypothesis.strategies as st
import torch

import beignet.metrics.functional.statistics
import beignet.statistics


@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=10),
    effect_size=st.floats(min_value=0.1, max_value=2.0),
    sample_size=st.integers(min_value=10, max_value=50),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_z_test_power(batch_size, effect_size, sample_size, alpha, dtype):
    """Test z_test_power functional wrapper."""
    effect_size_tensor = torch.full((batch_size,), effect_size, dtype=dtype)
    sample_size_tensor = torch.full((batch_size,), sample_size, dtype=torch.int64)

    result_functional = beignet.metrics.functional.statistics.z_test_power(
        effect_size_tensor,
        sample_size_tensor,
        alpha=alpha,
    )
    result_direct = beignet.statistics.z_test_power(
        effect_size_tensor,
        sample_size_tensor,
        alpha=alpha,
    )

    assert torch.allclose(result_functional, result_direct, atol=1e-6)
    assert torch.all(result_functional >= 0.0)
    assert torch.all(result_functional <= 1.0)
