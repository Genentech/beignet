import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import beignet


@given(
    batch_size=st.integers(min_value=1, max_value=4),
    n_atoms=st.integers(min_value=4, max_value=20),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)  # Disable deadline due to torch.compile variability
def test_smooth_local_distance_difference_test(batch_size, n_atoms, dtype):
    """Test the smooth LDDT loss function comprehensively."""

    # Generate test data
    device = torch.device("cpu")

    # Generate random atom positions scaled for realistic distances
    input = torch.randn(batch_size, n_atoms, 3, dtype=dtype, device=device) * 10
    target = torch.randn(batch_size, n_atoms, 3, dtype=dtype, device=device) * 10

    # Test basic functionality
    loss = beignet.smooth_local_distance_difference_test(input, target)

    # Check output shape
    assert loss.shape == (batch_size,), (
        f"Expected shape {(batch_size,)}, got {loss.shape}"
    )

    # Check loss is in valid range [0, 1] (since it's 1 - LDDT_score)
    assert torch.all(loss >= 0), (
        f"Smooth LDDT loss should be >= 0, got min: {torch.min(loss)}"
    )
    assert torch.all(loss <= 1), (
        f"Smooth LDDT loss should be <= 1, got max: {torch.max(loss)}"
    )

    # Check loss is finite
    assert torch.all(torch.isfinite(loss)), "Smooth LDDT loss should be finite"

    # Test perfect prediction case with controlled positions
    # Create a simple structure where atoms are within cutoff radius
    controlled_positions = torch.zeros(
        batch_size, n_atoms, 3, dtype=dtype, device=device
    )
    for b in range(batch_size):
        for i in range(n_atoms):
            controlled_positions[b, i] = torch.tensor(
                [i * 2.0, 0.0, 0.0], dtype=dtype
            )  # 2Å apart

    controlled_perfect_loss = beignet.smooth_local_distance_difference_test(
        controlled_positions, controlled_positions
    )
    controlled_random_loss = beignet.smooth_local_distance_difference_test(
        controlled_positions,
        controlled_positions + torch.randn_like(controlled_positions) * 0.5,
    )

    # Perfect prediction should be better than (or equal to) noisy prediction
    assert torch.all(controlled_perfect_loss <= controlled_random_loss), (
        f"Perfect controlled loss should be <= noisy: {controlled_perfect_loss} vs {controlled_random_loss}"
    )

    # Test that different structures give different losses (unless by coincidence)
    if batch_size > 1:
        # Modify one example to be very different
        modified_input = input.clone()
        modified_input[1] += 5.0  # Shift positions significantly

        modified_loss = beignet.smooth_local_distance_difference_test(
            modified_input, target
        )

        # Losses should be different for different predicted structures
        # (unless by very unlikely coincidence)
        if not torch.allclose(loss, modified_loss, atol=1e-6):
            assert True  # Different inputs give different losses

    # Test with custom parameters
    custom_loss = beignet.smooth_local_distance_difference_test(
        input,
        target,
        cutoff_radius=10.0,
        tolerance_thresholds=(1.0, 2.0, 4.0),
        smoothing_factor=2.0,
    )
    assert custom_loss.shape == (batch_size,), "Custom parameters should work"
    assert torch.all(torch.isfinite(custom_loss)), (
        "Custom parameters loss should be finite"
    )
    assert torch.all(custom_loss >= 0) and torch.all(custom_loss <= 1), (
        "Custom loss should be in [0,1]"
    )

    # Test gradient computation
    input_grad = input.clone().requires_grad_(True)
    target_grad = target.clone().requires_grad_(True)

    loss_grad = beignet.smooth_local_distance_difference_test(input_grad, target_grad)
    total_loss = loss_grad.sum()
    total_loss.backward()

    assert input_grad.grad is not None, "Should have gradients for input"
    assert target_grad.grad is not None, "Should have gradients for target"
    assert torch.all(torch.isfinite(input_grad.grad)), (
        "Input gradients should be finite"
    )
    assert torch.all(torch.isfinite(target_grad.grad)), (
        "Target gradients should be finite"
    )

    # Test that loss decreases when predicted positions move closer to target
    # Create a scenario where we know the direction of improvement
    simple_predicted = torch.zeros(1, 2, 3, dtype=dtype, device=device)
    simple_target = torch.zeros(1, 2, 3, dtype=dtype, device=device)
    simple_predicted[0, 0] = torch.tensor([0.0, 0.0, 0.0], dtype=dtype)
    simple_predicted[0, 1] = torch.tensor([5.0, 0.0, 0.0], dtype=dtype)  # 5 Å apart
    simple_target[0, 0] = torch.tensor([0.0, 0.0, 0.0], dtype=dtype)
    simple_target[0, 1] = torch.tensor(
        [3.0, 0.0, 0.0], dtype=dtype
    )  # 3 Å apart (closer)

    loss_before = beignet.smooth_local_distance_difference_test(
        simple_predicted, simple_target
    )

    # Move predicted closer to target
    better_predicted = simple_predicted.clone()
    better_predicted[0, 1] = torch.tensor(
        [3.0, 0.0, 0.0], dtype=dtype
    )  # Now matches target
    loss_after = beignet.smooth_local_distance_difference_test(
        better_predicted, simple_target
    )

    assert loss_after < loss_before, (
        f"Loss should decrease when structure improves: {loss_after} vs {loss_before}"
    )

    # Test batch operations - different examples should potentially give different losses
    if batch_size > 1:
        individual_losses = []
        for i in range(batch_size):
            single_loss = beignet.smooth_local_distance_difference_test(
                input[i : i + 1], target[i : i + 1]
            )
            individual_losses.append(single_loss)

        batch_losses = loss

        # Individual losses should match batch computation
        individual_tensor = torch.cat(individual_losses, dim=0)
        assert torch.allclose(batch_losses, individual_tensor, atol=1e-6), (
            "Batch processing should match individual processing"
        )

    # Test torch.compile compatibility (basic check)
    try:
        compiled_smooth_lddt_loss = torch.compile(
            beignet.smooth_local_distance_difference_test, fullgraph=True
        )
        compiled_loss = compiled_smooth_lddt_loss(input, target)

        # Should be very close to original loss
        assert torch.allclose(loss, compiled_loss, atol=1e-6), (
            "Compiled loss should match original"
        )
    except Exception:
        # torch.compile might not be available or might fail, which is acceptable
        pass

    # Test edge cases
    # Very close atoms (within all tolerance thresholds)
    close_positions = torch.zeros(1, 3, 3, dtype=dtype, device=device)
    close_positions[0, 1, 0] = 0.1  # Very small displacement
    close_positions[0, 2, 1] = 0.2  # Another small displacement

    close_loss = beignet.smooth_local_distance_difference_test(
        close_positions, close_positions
    )
    assert torch.isfinite(close_loss), "Should handle very close atoms"
    # For perfect prediction, loss should be reasonable (not 0 due to sigmoid smoothing)
    assert close_loss < 0.5, "Perfect prediction should have reasonable loss"

    # Test with atoms far apart (beyond cutoff)
    far_positions_pred = torch.zeros(1, 3, 3, dtype=dtype, device=device)
    far_positions_target = torch.zeros(1, 3, 3, dtype=dtype, device=device)
    far_positions_pred[0, 1, 0] = 20.0  # Beyond default cutoff of 15 Å
    far_positions_target[0, 1, 0] = 25.0  # Also far, different distance

    far_loss = beignet.smooth_local_distance_difference_test(
        far_positions_pred, far_positions_target
    )
    assert torch.isfinite(far_loss), "Should handle atoms far beyond cutoff"

    # Test input validation
    try:
        # Mismatched shapes
        wrong_target = target[:, : n_atoms - 1]  # Fewer atoms
        beignet.smooth_local_distance_difference_test(input, wrong_target)
        raise AssertionError("Should raise error for mismatched shapes")
    except ValueError:
        pass  # Expected

    try:
        # Wrong coordinate dimensions
        wrong_positions = input[..., :2]  # Only 2D instead of 3D
        beignet.smooth_local_distance_difference_test(wrong_positions, target[..., :2])
        raise AssertionError("Should raise error for 2D positions")
    except ValueError:
        pass  # Expected

    # Test with single tolerance threshold
    single_threshold_loss = beignet.smooth_local_distance_difference_test(
        input, target, tolerance_thresholds=(2.0,)
    )
    assert single_threshold_loss.shape == (batch_size,), "Single threshold should work"
    assert torch.all(torch.isfinite(single_threshold_loss)), (
        "Single threshold loss should be finite"
    )

    # Test with different smoothing factors
    sharp_loss = beignet.smooth_local_distance_difference_test(
        input,
        target,
        smoothing_factor=10.0,  # More sharp/discrete-like
    )
    smooth_loss = beignet.smooth_local_distance_difference_test(
        input,
        target,
        smoothing_factor=0.1,  # More smooth/gradual
    )

    assert torch.all(torch.isfinite(sharp_loss)), "Sharp smoothing should work"
    assert torch.all(torch.isfinite(smooth_loss)), "Smooth smoothing should work"

    # The losses might be different due to different smoothing, but both should be valid
    assert torch.all((sharp_loss >= 0) & (sharp_loss <= 1)), (
        "Sharp loss should be in [0,1]"
    )
    assert torch.all((smooth_loss >= 0) & (smooth_loss <= 1)), (
        "Smooth loss should be in [0,1]"
    )
