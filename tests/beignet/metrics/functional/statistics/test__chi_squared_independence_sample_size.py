import hypothesis
import hypothesis.strategies as st
import torch

import beignet.metrics.functional.statistics
import beignet.statistics


@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=10),
    effect_size=st.floats(min_value=0.1, max_value=2.0),
    power=st.floats(min_value=0.1, max_value=0.95),
    degrees_of_freedom=st.integers(min_value=1, max_value=10),
    alpha=st.floats(min_value=0.01, max_value=0.1),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_chi_squared_independence_sample_size(
    batch_size,
    effect_size,
    power,
    degrees_of_freedom,
    alpha,
    dtype,
):
    """Test chi_squared_independence_sample_size functional wrapper."""
    effect_size_tensor = torch.full((batch_size,), effect_size, dtype=dtype)
    power_tensor = torch.full((batch_size,), power, dtype=dtype)
    degrees_of_freedom_tensor = torch.full(
        (batch_size,),
        degrees_of_freedom,
        dtype=torch.int64,
    )

    result_functional = (
        beignet.metrics.functional.statistics.chi_squared_independence_sample_size(
            effect_size_tensor,
            power_tensor,
            degrees_of_freedom_tensor,
            alpha=alpha,
        )
    )

    result_direct = beignet.statistics.chi_squared_independence_sample_size(
        effect_size_tensor,
        power_tensor,
        degrees_of_freedom_tensor,
        alpha=alpha,
    )

    assert torch.allclose(result_functional, result_direct, atol=1e-6)
    assert torch.all(result_functional >= 1.0)
