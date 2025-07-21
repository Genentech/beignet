import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import beignet.nn.functional as F
from beignet.nn import DistogramLoss


@given(
    batch_size=st.integers(min_value=1, max_value=8),
    n_residues=st.integers(min_value=10, max_value=50),
    n_bins=st.integers(min_value=10, max_value=64),
    min_bin=st.floats(min_value=0.0, max_value=5.0),
    max_bin=st.floats(min_value=10.0, max_value=50.0),
    dtype=st.sampled_from([torch.float32, torch.float64]),
    reduction=st.sampled_from(["mean", "sum", "none"]),
    use_module=st.booleans(),
)
@settings(max_examples=50, deadline=None)
def test_distogram_loss(
    batch_size, n_residues, n_bins, min_bin, max_bin, dtype, reduction, use_module
):
    """Test distogram loss with comprehensive scenarios."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate predicted logits
    logits = torch.randn(
        batch_size, n_residues, n_residues, n_bins, dtype=dtype, device=device
    )

    # Generate target distances (ensure they're symmetric)
    target_distances = torch.rand(
        batch_size, n_residues, n_residues, dtype=dtype, device=device
    )
    target_distances = target_distances * (max_bin - min_bin) + min_bin
    # Make symmetric
    target_distances = (target_distances + target_distances.transpose(-2, -1)) / 2

    # Generate mask for valid residue pairs
    mask = torch.rand(batch_size, n_residues, n_residues, device=device) > 0.2
    mask = mask & mask.transpose(-2, -1)  # Ensure symmetry
    # Mask diagonal
    mask = mask & ~torch.eye(n_residues, dtype=torch.bool, device=device).unsqueeze(0)

    # Calculate loss
    if use_module:
        loss_fn = DistogramLoss(
            min_bin=min_bin, max_bin=max_bin, n_bins=n_bins, reduction=reduction
        )
        loss = loss_fn(logits, target_distances, mask)
    else:
        loss = F.distogram_loss(
            logits,
            target_distances,
            mask,
            min_bin=min_bin,
            max_bin=max_bin,
            n_bins=n_bins,
            reduction=reduction,
        )

    # Verify output shape
    if reduction == "none":
        assert loss.shape == (batch_size, n_residues, n_residues), (
            f"Expected shape {(batch_size, n_residues, n_residues)}, got {loss.shape}"
        )
    elif reduction == "sum":
        assert loss.shape == (), (
            f"Expected scalar output for sum reduction, got {loss.shape}"
        )
    else:  # mean
        assert loss.shape == (), (
            f"Expected scalar output for mean reduction, got {loss.shape}"
        )

    # Verify output is non-negative
    assert torch.all(loss >= 0), "Loss should be non-negative"

    # Verify output is finite
    assert torch.all(torch.isfinite(loss)), "Loss should be finite"

    # Test with all masked (should give zero loss with mean reduction)
    all_mask = torch.zeros_like(mask)
    if use_module:
        loss_all_masked = loss_fn(logits, target_distances, all_mask)
    else:
        loss_all_masked = F.distogram_loss(
            logits,
            target_distances,
            all_mask,
            min_bin=min_bin,
            max_bin=max_bin,
            n_bins=n_bins,
            reduction=reduction,
        )

    if reduction == "mean":
        assert loss_all_masked == 0.0, (
            "All masked loss should be zero with mean reduction"
        )

    # Test perfect prediction
    # Create logits that put all probability mass on the correct bin
    bin_edges = torch.linspace(min_bin, max_bin, n_bins + 1, dtype=dtype, device=device)
    bin_indices = torch.searchsorted(bin_edges[:-1], target_distances.contiguous()) - 1
    bin_indices = torch.clamp(bin_indices, 0, n_bins - 1)

    perfect_logits = torch.full_like(logits, -1e10)
    for b in range(batch_size):
        for i in range(n_residues):
            for j in range(n_residues):
                if mask[b, i, j]:
                    perfect_logits[b, i, j, bin_indices[b, i, j]] = 10.0

    if use_module:
        perfect_loss = loss_fn(perfect_logits, target_distances, mask)
    else:
        perfect_loss = F.distogram_loss(
            perfect_logits,
            target_distances,
            mask,
            min_bin=min_bin,
            max_bin=max_bin,
            n_bins=n_bins,
            reduction=reduction,
        )

    # Perfect prediction should have very low loss
    if reduction != "none":
        assert perfect_loss < 1e-4, (
            f"Perfect prediction loss {perfect_loss} should be near zero"
        )

    # Test gradient computation
    logits_grad = logits.clone().requires_grad_(True)

    if use_module:
        loss_grad = loss_fn(logits_grad, target_distances, mask)
    else:
        loss_grad = F.distogram_loss(
            logits_grad,
            target_distances,
            mask,
            min_bin=min_bin,
            max_bin=max_bin,
            n_bins=n_bins,
            reduction=reduction,
        )

    # Compute gradients
    if reduction == "none":
        loss_grad.sum().backward()
    else:
        loss_grad.backward()

    assert logits_grad.grad is not None, "Gradients should be computed for logits"
    assert torch.all(torch.isfinite(logits_grad.grad)), "Gradients should be finite"

    # Test torch.compile compatibility
    if n_residues < 30 and n_bins < 30:  # Avoid recompilation for large sizes
        compiled_fn = torch.compile(F.distogram_loss, fullgraph=True)
        loss_compiled = compiled_fn(
            logits,
            target_distances,
            mask,
            min_bin=min_bin,
            max_bin=max_bin,
            n_bins=n_bins,
            reduction=reduction,
        )

        assert torch.allclose(loss, loss_compiled, atol=1e-6, rtol=1e-6), (
            "Compiled function should produce same results"
        )

    # Test with torch.func transformations
    def compute_loss(logits_single, targets_single, mask_single):
        return F.distogram_loss(
            logits_single.unsqueeze(0),
            targets_single.unsqueeze(0),
            mask_single.unsqueeze(0),
            min_bin=min_bin,
            max_bin=max_bin,
            n_bins=n_bins,
            reduction="mean",
        )

    # Test vmap over batch dimension
    loss_vmap = torch.func.vmap(compute_loss)(logits, target_distances, mask)
    assert loss_vmap.shape == (batch_size,), "vmap should produce correct output shape"

    # Test dtype preservation
    assert loss.dtype == dtype, f"Output dtype should match input dtype {dtype}"

    # Test device preservation
    assert loss.device == device, f"Output device should match input device {device}"

    # Test bin edge computation
    bin_edges_test = torch.linspace(
        min_bin, max_bin, n_bins + 1, dtype=dtype, device=device
    )
    assert len(bin_edges_test) == n_bins + 1, "Should have n_bins + 1 edges"
    assert torch.allclose(bin_edges_test[0], torch.tensor(min_bin, dtype=dtype)), (
        "First edge should be min_bin"
    )
    assert torch.allclose(bin_edges_test[-1], torch.tensor(max_bin, dtype=dtype)), (
        "Last edge should be max_bin"
    )

    # Test symmetry invariance
    # The loss should be the same if we transpose logits and targets
    logits_T = logits.transpose(-3, -2)
    target_distances_T = target_distances.transpose(-2, -1)
    mask_T = mask.transpose(-2, -1)

    if use_module:
        loss_T = loss_fn(logits_T, target_distances_T, mask_T)
    else:
        loss_T = F.distogram_loss(
            logits_T,
            target_distances_T,
            mask_T,
            min_bin=min_bin,
            max_bin=max_bin,
            n_bins=n_bins,
            reduction=reduction,
        )

    if reduction != "none":
        assert torch.allclose(loss, loss_T, atol=1e-6, rtol=1e-6), (
            "Loss should be symmetric"
        )

    # Test error handling
    with pytest.raises(ValueError, match="min_bin must be less than max_bin"):
        F.distogram_loss(
            logits,
            target_distances,
            mask,
            min_bin=max_bin,
            max_bin=min_bin,
            n_bins=n_bins,
        )

    with pytest.raises(ValueError, match="n_bins must be at least 2"):
        F.distogram_loss(
            logits, target_distances, mask, min_bin=min_bin, max_bin=max_bin, n_bins=1
        )

    with pytest.raises(ValueError, match="Invalid reduction"):
        F.distogram_loss(
            logits,
            target_distances,
            mask,
            min_bin=min_bin,
            max_bin=max_bin,
            n_bins=n_bins,
            reduction="invalid",
        )

    # Test shape mismatch
    with pytest.raises(ValueError, match="Shape mismatch"):
        bad_targets = target_distances[..., : n_residues - 1]
        F.distogram_loss(
            logits, bad_targets, mask, min_bin=min_bin, max_bin=max_bin, n_bins=n_bins
        )

    # Test module parameters
    if use_module:
        assert loss_fn.min_bin == min_bin, "Module should store min_bin"
        assert loss_fn.max_bin == max_bin, "Module should store max_bin"
        assert loss_fn.n_bins == n_bins, "Module should store n_bins"
        assert loss_fn.reduction == reduction, "Module should store reduction"
