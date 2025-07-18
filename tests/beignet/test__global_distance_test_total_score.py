import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import beignet


@given(
    batch_size=st.integers(min_value=1, max_value=10),
    n_residues=st.integers(min_value=10, max_value=200),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)
def test_global_distance_test_total_score(batch_size, n_residues, dtype):
    # Set default dtype
    torch.set_default_dtype(dtype)

    # Test basic functionality
    input = torch.randn(batch_size, n_residues, 3)
    reference = torch.randn(batch_size, n_residues, 3)

    gdt_ts = beignet.global_distance_test_total_score(input, reference)

    # Check output shape
    assert gdt_ts.shape == (batch_size,)

    # Check output range
    assert torch.all(gdt_ts >= 0)
    assert torch.all(gdt_ts <= 1)

    # Test with identical structures (should give perfect score)
    identical_reference = input.clone()
    perfect_gdt_ts = beignet.global_distance_test_total_score(
        input, identical_reference
    )
    assert torch.allclose(perfect_gdt_ts, torch.ones(batch_size))

    # Test with mask
    mask = torch.rand(batch_size, n_residues) > 0.5
    masked_gdt_ts = beignet.global_distance_test_total_score(
        input, reference, mask=mask
    )
    assert masked_gdt_ts.shape == (batch_size,)
    assert torch.all(masked_gdt_ts >= 0)
    assert torch.all(masked_gdt_ts <= 1)

    # Test single structure (no batch)
    single_input = torch.randn(n_residues, 3)
    single_reference = torch.randn(n_residues, 3)
    single_gdt_ts = beignet.global_distance_test_total_score(
        single_input, single_reference
    )
    assert single_gdt_ts.shape == ()
    assert 0 <= single_gdt_ts <= 1

    # Test custom cutoffs
    custom_cutoffs = [0.5, 1.5, 3.0, 6.0]
    custom_gdt_ts = beignet.global_distance_test_total_score(
        input, reference, cutoffs=custom_cutoffs
    )
    assert custom_gdt_ts.shape == (batch_size,)

    # Test error cases
    with pytest.raises(ValueError, match="must have the same shape"):
        beignet.global_distance_test_total_score(torch.randn(10, 3), torch.randn(20, 3))

    with pytest.raises(ValueError, match="Mask shape"):
        beignet.global_distance_test_total_score(input, reference, mask=torch.ones(10))

    # Note: GDT scores use hard thresholds and are not differentiable in a meaningful way
    # The gradient would be zero almost everywhere except at the exact threshold boundaries

    # Test torch.compile compatibility
    compiled_fn = torch.compile(
        beignet.global_distance_test_total_score, fullgraph=True
    )
    compiled_result = compiled_fn(input, reference)
    assert torch.allclose(compiled_result, gdt_ts)

    # Test vmap compatibility
    vmapped_fn = torch.vmap(beignet.global_distance_test_total_score)
    vmap_input = torch.randn(5, batch_size, n_residues, 3)
    vmap_reference = torch.randn(5, batch_size, n_residues, 3)
    vmap_result = vmapped_fn(vmap_input, vmap_reference)
    assert vmap_result.shape == (5, batch_size)

    # Test edge case: all residues far apart
    far_reference = input + 1000.0
    far_gdt_ts = beignet.global_distance_test_total_score(input, far_reference)
    assert torch.allclose(far_gdt_ts, torch.zeros(batch_size))

    # Test edge case: empty mask
    empty_mask = torch.zeros(batch_size, n_residues, dtype=torch.bool)
    # This should handle division by zero gracefully
    empty_gdt_ts = beignet.global_distance_test_total_score(
        input, reference, mask=empty_mask
    )
    assert not torch.any(torch.isnan(empty_gdt_ts))

    # Reset default dtype
    torch.set_default_dtype(torch.float32)
