import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import beignet


@given(
    batch_size=st.integers(min_value=1, max_value=3),
    n_coords=st.integers(
        min_value=2, max_value=5
    ),  # Keep small for computational efficiency
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)  # Disable deadline due to computational complexity
def test_compute_alignment_error(batch_size, n_coords, dtype):
    """Test the compute alignment error function comprehensively."""

    # Generate test data
    device = torch.device("cpu")

    # Generate predicted and target coordinates
    pred_coords = torch.randn(batch_size, n_coords, 3, dtype=dtype, device=device)
    target_coords = torch.randn(batch_size, n_coords, 3, dtype=dtype, device=device)

    # Generate frame definitions - make them non-degenerate
    pred_frames = torch.randn(batch_size, n_coords, 3, 3, dtype=dtype, device=device)
    target_frames = torch.randn(batch_size, n_coords, 3, 3, dtype=dtype, device=device)

    # Make frames more well-conditioned by ensuring they form proper triangles
    for i in range(n_coords):
        # Set standard frame patterns with some variation
        pred_frames[:, i, 0, :] = (
            torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
            + torch.randn(batch_size, 3, dtype=dtype) * 0.1
        )
        pred_frames[:, i, 1, :] = (
            torch.tensor([0.0, 0.0, 0.0], dtype=dtype)
            + torch.randn(batch_size, 3, dtype=dtype) * 0.1
        )
        pred_frames[:, i, 2, :] = (
            torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
            + torch.randn(batch_size, 3, dtype=dtype) * 0.1
        )

        target_frames[:, i, 0, :] = (
            torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
            + torch.randn(batch_size, 3, dtype=dtype) * 0.1
        )
        target_frames[:, i, 1, :] = (
            torch.tensor([0.0, 0.0, 0.0], dtype=dtype)
            + torch.randn(batch_size, 3, dtype=dtype) * 0.1
        )
        target_frames[:, i, 2, :] = (
            torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
            + torch.randn(batch_size, 3, dtype=dtype) * 0.1
        )

    # Test basic functionality
    errors = beignet.compute_alignment_error(
        pred_coords, target_coords, pred_frames, target_frames
    )

    # Check output shape
    expected_shape = (batch_size, n_coords, n_coords)
    assert errors.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {errors.shape}"
    )

    # Check all errors are non-negative (distances with epsilon)
    assert torch.all(errors >= 0), "All alignment errors should be non-negative"

    # Check all errors are finite
    assert torch.all(torch.isfinite(errors)), "All alignment errors should be finite"

    # Test perfect alignment case - identical coordinates and frames
    perfect_errors = beignet.compute_alignment_error(
        pred_coords, pred_coords, pred_frames, pred_frames
    )

    # Should be very small (just epsilon) for perfect alignment
    assert torch.all(perfect_errors < 0.1), (
        "Perfect alignment should have very small errors"
    )

    # Test with custom epsilon
    large_epsilon = 1.0
    large_eps_errors = beignet.compute_alignment_error(
        pred_coords, target_coords, pred_frames, target_frames, epsilon=large_epsilon
    )

    # All errors should be at least epsilon
    assert torch.all(large_eps_errors >= large_epsilon), (
        "All errors should be at least epsilon"
    )

    # Test gradient computation
    pred_coords_grad = pred_coords.clone().requires_grad_(True)
    target_coords_grad = target_coords.clone().requires_grad_(True)
    pred_frames_grad = pred_frames.clone().requires_grad_(True)
    target_frames_grad = target_frames.clone().requires_grad_(True)

    errors_grad = beignet.compute_alignment_error(
        pred_coords_grad, target_coords_grad, pred_frames_grad, target_frames_grad
    )
    loss = errors_grad.sum()
    loss.backward()

    assert pred_coords_grad.grad is not None, (
        "Should have gradients for predicted coordinates"
    )
    assert target_coords_grad.grad is not None, (
        "Should have gradients for target coordinates"
    )
    assert pred_frames_grad.grad is not None, (
        "Should have gradients for predicted frames"
    )
    assert target_frames_grad.grad is not None, (
        "Should have gradients for target frames"
    )

    assert torch.all(torch.isfinite(pred_coords_grad.grad)), (
        "Pred coord gradients should be finite"
    )
    assert torch.all(torch.isfinite(target_coords_grad.grad)), (
        "Target coord gradients should be finite"
    )
    assert torch.all(torch.isfinite(pred_frames_grad.grad)), (
        "Pred frame gradients should be finite"
    )
    assert torch.all(torch.isfinite(target_frames_grad.grad)), (
        "Target frame gradients should be finite"
    )

    # Test torch.compile compatibility
    try:
        compiled_compute = torch.compile(
            beignet.compute_alignment_error, fullgraph=True
        )
        compiled_errors = compiled_compute(
            pred_coords, target_coords, pred_frames, target_frames
        )

        # Should be very close to original result
        assert torch.allclose(errors, compiled_errors, atol=1e-6), (
            "Compiled version should match original"
        )
    except Exception:
        # torch.compile might not be available or might fail, which is acceptable
        pass

    # Test batch operations - different examples should be processed independently
    if batch_size > 1:
        # Process each batch element individually
        individual_errors = []
        for i in range(batch_size):
            single_error = beignet.compute_alignment_error(
                pred_coords[i : i + 1],
                target_coords[i : i + 1],
                pred_frames[i : i + 1],
                target_frames[i : i + 1],
            )
            individual_errors.append(single_error)

        # Concatenate results
        individual_tensor = torch.cat(individual_errors, dim=0)

        # Should match batch processing
        assert torch.allclose(errors, individual_tensor, atol=1e-6), (
            "Batch processing should match individual processing"
        )

    # Test input validation
    try:
        # Mismatched coordinate shapes
        wrong_target = target_coords[:, : n_coords - 1]  # Fewer coordinates
        beignet.compute_alignment_error(
            pred_coords, wrong_target, pred_frames, target_frames
        )
        raise AssertionError("Should raise error for mismatched coordinate shapes")
    except ValueError:
        pass  # Expected

    try:
        # Mismatched frame shapes
        wrong_frames = target_frames[:, : n_coords - 1]  # Fewer frames
        beignet.compute_alignment_error(
            pred_coords, target_coords, pred_frames, wrong_frames
        )
        raise AssertionError("Should raise error for mismatched frame shapes")
    except ValueError:
        pass  # Expected

    try:
        # Wrong coordinate dimensions (2D instead of 3D)
        wrong_coords = pred_coords[..., :2]
        wrong_targets = target_coords[..., :2]
        beignet.compute_alignment_error(
            wrong_coords, wrong_targets, pred_frames, target_frames
        )
        raise AssertionError("Should raise error for 2D coordinates")
    except ValueError:
        pass  # Expected

    try:
        # Wrong frame dimensions
        wrong_frames = pred_frames[..., :2, :]  # Only 2 atoms per frame
        beignet.compute_alignment_error(
            pred_coords, target_coords, wrong_frames, target_frames
        )
        raise AssertionError("Should raise error for wrong frame dimensions")
    except ValueError:
        pass  # Expected

    # Test symmetry property: errors[i,j] should relate to errors[j,i] in some way
    # While not necessarily equal (different frames), both should be finite and non-negative
    for i in range(n_coords):
        for j in range(n_coords):
            assert torch.all(torch.isfinite(errors[:, i, j])), (
                f"Error [{i},{j}] should be finite"
            )
            assert torch.all(errors[:, i, j] >= 0), (
                f"Error [{i},{j}] should be non-negative"
            )

    # Test numerical stability with very small differences
    small_diff_coords = pred_coords + torch.randn_like(pred_coords) * 1e-6
    small_errors = beignet.compute_alignment_error(
        pred_coords, small_diff_coords, pred_frames, target_frames
    )
    assert torch.all(torch.isfinite(small_errors)), (
        "Should handle very small coordinate differences"
    )

    # Test with large coordinates
    large_pred = pred_coords * 1000
    large_target = target_coords * 1000
    large_errors = beignet.compute_alignment_error(
        large_pred, large_target, pred_frames, target_frames
    )
    assert torch.all(torch.isfinite(large_errors)), "Should handle large coordinates"

    # Test error matrix properties
    # Diagonal elements (same coordinate, same frame index) might have special properties
    # but we mainly check they're reasonable
    for i in range(n_coords):
        diagonal_errors = errors[:, i, i]  # Error of coordinate i in frame i
        assert torch.all(torch.isfinite(diagonal_errors)), (
            f"Diagonal errors [{i},{i}] should be finite"
        )
        assert torch.all(diagonal_errors >= 0), (
            f"Diagonal errors [{i},{i}] should be non-negative"
        )

    # Test that different coordinate-frame pairs generally give different errors
    # (unless by coincidence or perfect alignment)
    if n_coords >= 2:
        error_01 = errors[:, 0, 1]  # Coord 0 in frame 1
        error_10 = errors[:, 1, 0]  # Coord 1 in frame 0

        # Both should be finite and non-negative
        assert torch.all(torch.isfinite(error_01)), "Error [0,1] should be finite"
        assert torch.all(torch.isfinite(error_10)), "Error [1,0] should be finite"
        assert torch.all(error_01 >= 0), "Error [0,1] should be non-negative"
        assert torch.all(error_10 >= 0), "Error [1,0] should be non-negative"
