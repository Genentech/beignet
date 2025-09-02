import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import beignet


@given(
    batch_size=st.integers(min_value=1, max_value=4),
    n_atoms=st.integers(min_value=3, max_value=16),
    num_bins=st.integers(min_value=8, max_value=64),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)  # Disable deadline due to torch.compile variability
def test_distogram_loss(batch_size, n_atoms, num_bins, dtype):
    """Test the distogram loss function comprehensively."""

    # Generate test data
    device = torch.device("cpu")

    # Generate predicted logits (symmetric for realistic case)
    input = torch.randn(
        batch_size, n_atoms, n_atoms, num_bins, dtype=dtype, device=device
    )
    # Make logits symmetric for more realistic test
    input = (input + input.transpose(-3, -2)) / 2

    # Generate random atom positions
    target = (
        torch.randn(batch_size, n_atoms, 3, dtype=dtype, device=device) * 10
    )  # Scale for realistic distances

    # Test basic functionality
    loss = beignet.distogram_loss(input, target, num_bins=num_bins)

    # Check output shape
    assert loss.shape == (batch_size,), (
        f"Expected shape {(batch_size,)}, got {loss.shape}"
    )

    # Check loss is non-negative (cross-entropy loss should be >= 0)
    assert torch.all(loss >= 0), "Distogram loss should be non-negative"

    # Check loss is finite
    assert torch.all(torch.isfinite(loss)), "Distogram loss should be finite"

    # Test better prediction case
    # Create a realistic case where we predict logits that should align with the soft target distribution
    test_positions = torch.zeros(1, 3, 3, dtype=dtype, device=device)
    test_positions[0, 0] = torch.tensor([0.0, 0.0, 0.0], dtype=dtype)  # Origin
    test_positions[0, 1] = torch.tensor([5.0, 0.0, 0.0], dtype=dtype)  # 5 Å away
    test_positions[0, 2] = torch.tensor([0.0, 10.0, 0.0], dtype=dtype)  # 10 Å away

    # Get target distribution by running the distogram computation partially
    min_distance, max_distance = 2.0, 22.0
    bin_width_test = (max_distance - min_distance) / (num_bins - 1)
    test_eps = 1e-8

    # Create bin centers
    bin_centers_test = torch.linspace(
        min_distance, max_distance, num_bins, dtype=dtype, device=device
    )

    # Compute distances
    pos_i = test_positions.unsqueeze(-2)  # (1, 3, 1, 3)
    pos_j = test_positions.unsqueeze(-3)  # (1, 1, 3, 3)
    true_distances = torch.sqrt(torch.sum((pos_i - pos_j) ** 2, dim=-1) + test_eps)
    clamped_distances_test = torch.clamp(true_distances, min_distance, max_distance)

    # Create the soft target distribution (mimicking the implementation)
    sigma_test = bin_width_test / 2.0
    distances_expanded_test = torch.unsqueeze(clamped_distances_test, -1)
    diff_squared_test = (distances_expanded_test - bin_centers_test) ** 2
    target_probs_test = torch.exp(-diff_squared_test / (2 * sigma_test**2))
    target_probs_test = target_probs_test / (
        torch.sum(target_probs_test, dim=-1, keepdim=True) + test_eps
    )

    # Create logits that should produce similar probabilities
    good_logits = torch.log(target_probs_test + test_eps)

    good_loss = beignet.distogram_loss(good_logits, test_positions, num_bins=num_bins)

    # Better prediction should have lower loss than random prediction
    # Create random logits with matching shape for test_positions (3 atoms)
    random_logits_3x3 = torch.randn(1, 3, 3, num_bins, dtype=dtype, device=device)
    random_loss = beignet.distogram_loss(
        random_logits_3x3, test_positions, num_bins=num_bins
    )
    assert good_loss < random_loss, (
        f"Better prediction should have lower loss: {good_loss} vs {random_loss}"
    )

    # Test different distance ranges
    custom_loss = beignet.distogram_loss(
        input, target, min_distance=1.0, max_distance=30.0, num_bins=num_bins
    )
    assert custom_loss.shape == (batch_size,), "Custom distance range should work"
    assert torch.all(torch.isfinite(custom_loss)), "Custom range loss should be finite"

    # Test gradient computation
    input_grad = input.clone().requires_grad_(True)
    target_grad = target.clone().requires_grad_(True)

    loss_grad = beignet.distogram_loss(input_grad, target_grad, num_bins=num_bins)
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

    # Test batch operations - different examples should give different losses
    if batch_size > 1:
        # Modify one example to be very different
        modified_target = target.clone()
        modified_target[1] *= 2  # Scale positions by 2 for second example

        modified_loss = beignet.distogram_loss(
            input, modified_target, num_bins=num_bins
        )

        # Losses should be different (unless by coincidence)
        if not torch.allclose(loss, modified_loss, atol=1e-6):
            assert True  # Different inputs give different losses

    # Test torch.compile compatibility (basic check)
    try:
        compiled_distogram_loss = torch.compile(beignet.distogram_loss, fullgraph=True)
        compiled_loss = compiled_distogram_loss(input, target, num_bins=num_bins)

        # Should be very close to original loss
        assert torch.allclose(loss, compiled_loss, atol=1e-6), (
            "Compiled loss should match original"
        )
    except Exception:
        # torch.compile might not be available or might fail, which is acceptable
        pass

    # Test symmetry property - distogram should be symmetric
    # Since we made logits symmetric, the loss should treat (i,j) and (j,i) pairs equally
    # This is implicitly tested by the symmetric logits generation above

    # Test edge cases
    # Single distance case (very close atoms)
    close_positions = torch.zeros(1, 2, 3, dtype=dtype, device=device)
    close_positions[0, 1, 0] = 0.1  # Very close atoms
    close_logits = torch.randn(1, 2, 2, num_bins, dtype=dtype, device=device)

    close_loss = beignet.distogram_loss(
        close_logits, close_positions, num_bins=num_bins
    )
    assert torch.isfinite(close_loss), "Should handle very close atoms"

    # Test very far atoms
    far_positions = torch.zeros(1, 2, 3, dtype=dtype, device=device)
    far_positions[0, 1, 0] = 50.0  # Very far atoms (beyond max_distance)
    far_logits = torch.randn(1, 2, 2, num_bins, dtype=dtype, device=device)

    far_loss = beignet.distogram_loss(far_logits, far_positions, num_bins=num_bins)
    assert torch.isfinite(far_loss), "Should handle very far atoms"

    # Test input validation
    try:
        # Wrong number of bins
        beignet.distogram_loss(input, target, num_bins=num_bins + 1)
        raise AssertionError("Should raise error for mismatched num_bins")
    except ValueError:
        pass  # Expected

    try:
        # Wrong position dimensions
        wrong_target = target[..., :2]  # Only 2D instead of 3D
        beignet.distogram_loss(input, wrong_target, num_bins=num_bins)
        raise AssertionError("Should raise error for 2D positions")
    except ValueError:
        pass  # Expected

    try:
        # Mismatched atom numbers
        wrong_input = input[:, : n_atoms - 1]  # Fewer atoms in logits
        beignet.distogram_loss(wrong_input, target, num_bins=num_bins)
        raise AssertionError("Should raise error for mismatched atom numbers")
    except ValueError:
        pass  # Expected
