import hypothesis
import hypothesis.strategies as st
import torch

import beignet.metrics.functional.statistics
import beignet.statistics


@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=10),
    sample_size_per_group=st.integers(min_value=5, max_value=20),
    num_groups=st.integers(min_value=3, max_value=6),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@hypothesis.settings(deadline=None)
def test_cohens_f_squared(batch_size, sample_size_per_group, num_groups, dtype):
    """Test cohens_f_squared functional wrapper."""
    groups = []
    for i in range(num_groups):
        group = torch.randn(batch_size, sample_size_per_group, dtype=dtype) + float(i)
        groups.append(group)

    result_functional = beignet.metrics.functional.statistics.cohens_f_squared(groups)

    # Compare with Cohen's f squared - manually compute between-groups and within-groups standard deviations
    group_means = torch.stack([group.mean(dim=-1) for group in groups], dim=-1)

    sample_sizes = torch.tensor(
        [group.shape[-1] for group in groups],
        dtype=group_means.dtype,
        device=group_means.device,
    )
    total_sample_size = sample_sizes.sum()
    weights = sample_sizes / total_sample_size
    overall_mean = (group_means * weights).sum(dim=-1, keepdim=True)

    between_groups_variance = ((group_means - overall_mean) ** 2 * weights).sum(dim=-1)
    between_groups_std = torch.sqrt(between_groups_variance)

    within_groups_variances = []
    for group in groups:
        within_var = torch.var(group, dim=-1, unbiased=True)
        within_groups_variances.append(within_var)

    dofs = torch.tensor(
        [group.shape[-1] - 1 for group in groups],
        dtype=group_means.dtype,
        device=group_means.device,
    )
    total_dof = dofs.sum()

    pooled_within_variance = (
        sum(dof * var for dof, var in zip(dofs, within_groups_variances, strict=False))
        / total_dof
    )
    pooled_within_std = torch.sqrt(pooled_within_variance)

    result_direct = beignet.statistics.cohens_f_squared(
        between_groups_std,
        pooled_within_std,
    )

    assert torch.allclose(result_functional, result_direct, atol=1e-6)
    assert result_functional.shape == (batch_size,)
    assert result_functional.dtype == dtype
    assert torch.all(result_functional >= 0.0)
