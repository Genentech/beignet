import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import beignet


@given(
    batch_size=st.integers(min_value=1, max_value=4),
    n_frames=st.integers(min_value=1, max_value=8),
    n_atoms=st.integers(min_value=1, max_value=10),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None, max_examples=3)
def test_frame_aligned_point_error(batch_size, n_frames, n_atoms, dtype):
    """Test FAPE loss computation with various input configurations."""
    device = torch.device("cpu")

    # Generate random frames (rotations should be orthogonal matrices)
    frame_rot = torch.randn(batch_size, n_frames, 3, 3, dtype=dtype, device=device)
    frame_rot = torch.linalg.qr(frame_rot).Q  # Ensure orthogonal matrices
    frame_trans = torch.randn(batch_size, n_frames, 3, dtype=dtype, device=device)

    target_rot = torch.randn(batch_size, n_frames, 3, 3, dtype=dtype, device=device)
    target_rot = torch.linalg.qr(target_rot).Q  # Ensure orthogonal matrices
    target_trans = torch.randn(batch_size, n_frames, 3, dtype=dtype, device=device)

    # Generate random atom positions
    pos = torch.randn(batch_size, n_atoms, 3, dtype=dtype, device=device)
    target_pos = torch.randn(batch_size, n_atoms, 3, dtype=dtype, device=device)

    # Test basic functionality
    loss = beignet.frame_aligned_point_error(
        frame_rot, frame_trans, pos, target_rot, target_trans, target_pos
    )

    # Test output shape and properties
    assert loss.shape == (batch_size,), (
        f"Expected shape {(batch_size,)}, got {loss.shape}"
    )
    assert torch.all(loss >= 0), "FAPE loss should be non-negative"
    assert torch.all(torch.isfinite(loss)), "FAPE loss should be finite"

    # Test perfect prediction (identical frames and positions)
    zero_loss = beignet.frame_aligned_point_error(
        frame_rot, frame_trans, pos, frame_rot, frame_trans, pos
    )
    assert torch.allclose(zero_loss, torch.zeros_like(zero_loss), atol=1e-6), (
        "FAPE loss should be zero for identical predictions"
    )

    # Test unclamped vs clamped loss
    unclamped_loss = beignet.frame_aligned_point_error(
        frame_rot,
        frame_trans,
        pos,
        target_rot,
        target_trans,
        target_pos,
        clamp_distance=None,
    )
    clamped_loss = beignet.frame_aligned_point_error(
        frame_rot,
        frame_trans,
        pos,
        target_rot,
        target_trans,
        target_pos,
        clamp_distance=5.0,
    )

    assert torch.all(clamped_loss <= unclamped_loss + 1e-6), (
        "Clamped loss should be less than or equal to unclamped loss"
    )

    # Test different clamp values
    large_clamp_loss = beignet.frame_aligned_point_error(
        frame_rot,
        frame_trans,
        pos,
        target_rot,
        target_trans,
        target_pos,
        clamp_distance=100.0,
    )
    small_clamp_loss = beignet.frame_aligned_point_error(
        frame_rot,
        frame_trans,
        pos,
        target_rot,
        target_trans,
        target_pos,
        clamp_distance=1.0,
    )

    assert torch.all(small_clamp_loss <= large_clamp_loss + 1e-6), (
        "Smaller clamp should result in smaller or equal loss"
    )

    # Test gradient computation
    frame_rot_grad = frame_rot.clone().requires_grad_(True)
    frame_trans_grad = frame_trans.clone().requires_grad_(True)
    pos_grad = pos.clone().requires_grad_(True)

    loss_grad = beignet.frame_aligned_point_error(
        frame_rot_grad, frame_trans_grad, pos_grad, target_rot, target_trans, target_pos
    )
    total_loss = torch.sum(loss_grad)
    total_loss.backward()

    assert frame_rot_grad.grad is not None, "Rotation gradients should be computed"
    assert frame_trans_grad.grad is not None, "Translation gradients should be computed"
    assert pos_grad.grad is not None, "Position gradients should be computed"
    assert torch.all(torch.isfinite(frame_rot_grad.grad)), (
        "Rotation gradients should be finite"
    )
    assert torch.all(torch.isfinite(frame_trans_grad.grad)), (
        "Translation gradients should be finite"
    )
    assert torch.all(torch.isfinite(pos_grad.grad)), (
        "Position gradients should be finite"
    )

    # Test torch.compile compatibility
    compiled_fape = torch.compile(beignet.frame_aligned_point_error, fullgraph=True)
    compiled_loss = compiled_fape(
        frame_rot, frame_trans, pos, target_rot, target_trans, target_pos
    )

    assert torch.allclose(loss, compiled_loss, atol=1e-6), (
        "Compiled and non-compiled versions should give same results"
    )

    # Test batch operations work correctly
    if batch_size > 1:
        # Split batch and compute individually
        individual_losses = []
        for i in range(batch_size):
            frame_rot_i = frame_rot[i : i + 1]
            frame_trans_i = frame_trans[i : i + 1]
            target_rot_i = target_rot[i : i + 1]
            target_trans_i = target_trans[i : i + 1]
            pos_i = pos[i : i + 1]
            target_pos_i = target_pos[i : i + 1]

            loss_i = beignet.frame_aligned_point_error(
                frame_rot_i,
                frame_trans_i,
                pos_i,
                target_rot_i,
                target_trans_i,
                target_pos_i,
            )
            individual_losses.append(loss_i)

        individual_stack = torch.cat(individual_losses, dim=0)
        assert torch.allclose(loss, individual_stack, atol=1e-6), (
            "Batched computation should match individual computations"
        )

    # Test error handling for mismatched shapes
    try:
        wrong_rot = torch.randn(
            batch_size, n_frames + 1, 3, 3, dtype=dtype, device=device
        )
        beignet.frame_aligned_point_error(
            wrong_rot, frame_trans, pos, target_rot, target_trans, target_pos
        )
        raise AssertionError("Should raise error for mismatched frame shapes")
    except ValueError:
        pass  # Expected

    try:
        wrong_pos = torch.randn(batch_size, n_atoms + 1, 3, dtype=dtype, device=device)
        beignet.frame_aligned_point_error(
            frame_rot, frame_trans, wrong_pos, target_rot, target_trans, target_pos
        )
        raise AssertionError("Should raise error for mismatched position shapes")
    except ValueError:
        pass  # Expected

    # Test numerical stability (eps is now handled internally)
    stable_loss = beignet.frame_aligned_point_error(
        frame_rot, frame_trans, pos, target_rot, target_trans, target_pos
    )

    # Should be stable and finite
    assert torch.all(torch.isfinite(stable_loss)), "Loss should be numerically stable"

    # Test torch.func transformations
    def fape_wrapper(frame_rot, frame_trans, pos, target_rot, target_trans, target_pos):
        return beignet.frame_aligned_point_error(
            frame_rot, frame_trans, pos, target_rot, target_trans, target_pos
        )

    # Test vmap over batch dimension
    if batch_size > 1:
        vmapped_fape = torch.vmap(fape_wrapper, in_dims=(0, 0, 0, 0, 0, 0))

        vmapped_loss = vmapped_fape(
            frame_rot, frame_trans, pos, target_rot, target_trans, target_pos
        )

        assert torch.allclose(loss, vmapped_loss, atol=1e-6), (
            "vmap should give same results as batched computation"
        )
