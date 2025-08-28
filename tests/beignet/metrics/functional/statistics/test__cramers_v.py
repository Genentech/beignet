import hypothesis
import hypothesis.strategies as st
import torch

import beignet.metrics.functional.statistics
import beignet.statistics


@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=10),
    n_rows=st.integers(min_value=2, max_value=5),
    n_cols=st.integers(min_value=2, max_value=5),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_cramers_v(batch_size, n_rows, n_cols, dtype):
    """Test cramers_v functional wrapper."""
    contingency_table = torch.randint(
        1,
        50,
        (batch_size, n_rows, n_cols),
        dtype=torch.int64,
    ).to(dtype)

    result_functional = beignet.metrics.functional.statistics.cramers_v(
        contingency_table,
    )

    # Manually compute chi-square, sample size, and min_dim to compare with statistics function
    row_totals = contingency_table.sum(dim=-1, keepdim=True)
    col_totals = contingency_table.sum(dim=-2, keepdim=True)
    sample_size = contingency_table.sum(dim=(-2, -1), keepdim=True)

    expected = (row_totals * col_totals) / sample_size
    chi_square = torch.sum(
        (contingency_table - expected) ** 2
        / torch.clamp(expected, min=torch.finfo(contingency_table.dtype).eps),
        dim=(-2, -1),
    )

    sample_size_flat = sample_size.squeeze((-2, -1))
    min_dim = torch.tensor(
        min(n_rows, n_cols) - 1,
        dtype=dtype,
        device=contingency_table.device,
    )
    min_dim = min_dim.expand(chi_square.shape)

    result_direct = beignet.statistics.cramers_v(chi_square, sample_size_flat, min_dim)

    assert torch.allclose(result_functional, result_direct, atol=1e-6)
    assert torch.all(result_functional >= 0.0)
    assert torch.all(result_functional <= 1.0)
