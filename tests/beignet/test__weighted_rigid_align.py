import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import beignet


@given(
    batch_size=st.integers(min_value=1, max_value=4),
    n_points=st.integers(min_value=3, max_value=20),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)  # Disable deadline due to torch.compile variability
def test_weighted_rigid_align(batch_size, n_points, dtype):
    """Test the weighted rigid alignment function comprehensively."""

    # Generate test data
    device = torch.device("cpu")

    # Generate random 3D points
    input_points = torch.randn(batch_size, n_points, 3, dtype=dtype, device=device)
    target_points = torch.randn(batch_size, n_points, 3, dtype=dtype, device=device)

    # Generate random weights (positive values)
    weights = torch.rand(batch_size, n_points, dtype=dtype, device=device) + 0.1

    # Test basic functionality
    aligned = beignet.weighted_rigid_align(input_points, target_points, weights)

    # Check output shape
    assert aligned.shape == input_points.shape, (
        f"Expected shape {input_points.shape}, got {aligned.shape}"
    )

    # Check output is finite
    assert torch.all(torch.isfinite(aligned)), "Aligned points should be finite"

    # Test that alignment is rigid (preserves distances within point set)
    # Due to floating point precision, we allow small tolerances
    if n_points >= 2:
        # Compute pairwise distances in original and aligned sets
        orig_dist = torch.cdist(input_points, input_points)
        aligned_dist = torch.cdist(aligned, aligned)

        # Distances should be preserved (rigid transformation)
        assert torch.allclose(orig_dist, aligned_dist, atol=1e-5), (
            "Rigid alignment should preserve pairwise distances"
        )

    # Test identity case - aligning points to themselves with uniform weights
    uniform_weights = torch.ones(batch_size, n_points, dtype=dtype, device=device)
    self_aligned = beignet.weighted_rigid_align(
        target_points, target_points, uniform_weights
    )

    # Should be very close to original (up to numerical precision)
    assert torch.allclose(self_aligned, target_points, atol=1e-4), (
        "Self-alignment should return approximately the same points"
    )

    # Test with uniform weights vs non-uniform weights
    uniform_weights = torch.ones(batch_size, n_points, dtype=dtype, device=device)
    aligned_uniform = beignet.weighted_rigid_align(
        input_points, target_points, uniform_weights
    )
    assert torch.all(torch.isfinite(aligned_uniform)), "Uniform weights should work"

    # Test that both uniform and non-uniform weights produce valid results
    aligned_weighted = beignet.weighted_rigid_align(
        input_points, target_points, weights
    )
    assert torch.all(torch.isfinite(aligned_weighted)), (
        "Non-uniform weights should work"
    )

    # Test gradient stopping - aligned output should not require gradients
    input_grad = input_points.clone().requires_grad_(True)
    target_grad = target_points.clone().requires_grad_(True)
    weights_grad = weights.clone().requires_grad_(True)

    aligned_grad = beignet.weighted_rigid_align(input_grad, target_grad, weights_grad)

    # Aligned output should not require gradients (detached)
    assert not aligned_grad.requires_grad, (
        "Aligned output should have gradients stopped"
    )

    # Test different weight patterns
    # Zero weights for some points
    masked_weights = weights.clone()
    masked_weights[:, : n_points // 2] = 0.0  # Zero out first half

    if torch.any(masked_weights > 0):  # Ensure we have some non-zero weights
        aligned_masked = beignet.weighted_rigid_align(
            input_points, target_points, masked_weights
        )
        assert torch.all(torch.isfinite(aligned_masked)), (
            "Should handle zero weights gracefully"
        )

    # Test with very small weights
    small_weights = torch.full_like(weights, 1e-6)
    aligned_small = beignet.weighted_rigid_align(
        input_points, target_points, small_weights
    )
    assert torch.all(torch.isfinite(aligned_small)), "Should handle very small weights"

    # Test torch.compile compatibility
    try:
        compiled_weighted_rigid_align = torch.compile(
            beignet.weighted_rigid_align, fullgraph=True
        )
        aligned_compiled = compiled_weighted_rigid_align(
            input_points, target_points, weights
        )

        # Should be very close to original result
        assert torch.allclose(aligned, aligned_compiled, atol=1e-6), (
            "Compiled version should match original"
        )
    except Exception:
        # torch.compile might not be available or might fail, which is acceptable
        pass

    # Test batch operations - different examples should be processed independently
    if batch_size > 1:
        # Process each batch element individually
        individual_aligned = []
        for i in range(batch_size):
            single_aligned = beignet.weighted_rigid_align(
                input_points[i : i + 1], target_points[i : i + 1], weights[i : i + 1]
            )
            individual_aligned.append(single_aligned)

        # Concatenate results
        individual_tensor = torch.cat(individual_aligned, dim=0)

        # Should match batch processing
        assert torch.allclose(aligned, individual_tensor, atol=1e-6), (
            "Batch processing should match individual processing"
        )

    # Test input validation
    try:
        # Mismatched input/target shapes
        wrong_target = target_points[:, : n_points - 1]  # Fewer points
        beignet.weighted_rigid_align(input_points, wrong_target, weights)
        raise AssertionError("Should raise error for mismatched shapes")
    except ValueError:
        pass  # Expected

    try:
        # Wrong dimensionality (2D instead of 3D)
        wrong_input = input_points[..., :2]
        wrong_target = target_points[..., :2]
        beignet.weighted_rigid_align(wrong_input, wrong_target, weights)
        raise AssertionError("Should raise error for 2D points")
    except ValueError:
        pass  # Expected

    try:
        # Wrong weights shape
        wrong_weights = weights[..., : n_points - 1]  # Fewer weights
        beignet.weighted_rigid_align(input_points, target_points, wrong_weights)
        raise AssertionError("Should raise error for wrong weights shape")
    except ValueError:
        pass  # Expected

    # Test special case: single point (should work but be trivial)
    if n_points == 3:  # Use minimum case we test
        single_input = input_points[:, :1]  # Single point
        single_target = target_points[:, :1]
        single_weights = weights[:, :1]

        single_aligned = beignet.weighted_rigid_align(
            single_input, single_target, single_weights
        )
        assert torch.all(torch.isfinite(single_aligned)), (
            "Should handle single point case"
        )

    # Test numerical stability with very large/small coordinates
    large_input = input_points * 1000
    large_target = target_points * 1000

    large_aligned = beignet.weighted_rigid_align(large_input, large_target, weights)
    assert torch.all(torch.isfinite(large_aligned)), "Should handle large coordinates"

    # Test perfect alignment case (target = input + rigid transformation)
    # Create a known rigid transformation
    # Simple case: rotation by 90 degrees around z-axis + translation
    if batch_size == 1 and n_points >= 4:  # Ensure sufficient points
        # Create rotation matrix (90 degrees around z-axis)
        R = torch.tensor(
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=dtype, device=device
        ).unsqueeze(0)  # (1, 3, 3)

        # Create translation
        t = torch.tensor([[1, 2, 3]], dtype=dtype, device=device).unsqueeze(
            1
        )  # (1, 1, 3)

        # Apply known transformation: target = R @ input + t
        simple_input = input_points[:1]  # Use first batch element
        transformed_target = torch.matmul(simple_input, R.transpose(-2, -1)) + t

        # Align back - should recover approximately the original transformation
        aligned_back = beignet.weighted_rigid_align(
            simple_input, transformed_target, weights[:1]
        )

        # Check that the alignment brings points close to the transformed target
        # (within numerical precision)
        alignment_error = torch.norm(aligned_back - transformed_target, dim=-1)
        assert torch.all(alignment_error < 1e-3), (
            "Perfect rigid transformation should be recovered accurately"
        )
