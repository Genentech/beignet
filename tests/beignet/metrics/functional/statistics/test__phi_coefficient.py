import hypothesis
import hypothesis.strategies as st
import torch

import beignet.metrics.functional.statistics
import beignet.statistics


@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=10),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_phi_coefficient(batch_size, dtype):
    """Test phi_coefficient functional wrapper."""
    contingency_table = torch.randint(1, 50, (batch_size, 2, 2), dtype=torch.int64).to(
        dtype,
    )

    result_functional = beignet.metrics.functional.statistics.phi_coefficient(
        contingency_table,
    )

    # Manually compute chi-square and sample size to compare with statistics function
    a = contingency_table[..., 0, 0]
    b = contingency_table[..., 0, 1]
    c = contingency_table[..., 1, 0]
    d = contingency_table[..., 1, 1]

    sample_size = contingency_table.sum(dim=(-2, -1))

    numerator = sample_size * (a * d - b * c) ** 2
    denominator = (a + b) * (c + d) * (a + c) * (b + d)
    denominator = torch.clamp(denominator, min=torch.finfo(contingency_table.dtype).eps)
    chi_square = numerator / denominator

    result_direct = beignet.statistics.phi_coefficient(chi_square, sample_size)

    assert torch.allclose(result_functional, result_direct, atol=1e-6)
    assert torch.all(result_functional >= -1.0)
    assert torch.all(result_functional <= 1.0)
