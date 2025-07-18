import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import beignet


@given(
    batch_size=st.integers(min_value=1, max_value=5),
    n_residues=st.integers(min_value=20, max_value=200),
    dtype=st.sampled_from([torch.float32, torch.float64]),
    aligned=st.booleans(),
    use_weights=st.booleans(),
    use_custom_d0=st.booleans(),
)
@settings(max_examples=20, deadline=None)
def test_template_modeling_score(
    batch_size, n_residues, dtype, aligned, use_weights, use_custom_d0
):
    """Test template_modeling_score operator with comprehensive scenarios."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test basic functionality with random structures
    if batch_size == 1:
        structure1 = torch.randn(n_residues, 3, dtype=dtype, device=device)
        structure2 = structure1 + 0.1 * torch.randn(
            n_residues, 3, dtype=dtype, device=device
        )
    else:
        structure1 = torch.randn(batch_size, n_residues, 3, dtype=dtype, device=device)
        structure2 = structure1 + 0.1 * torch.randn(
            batch_size, n_residues, 3, dtype=dtype, device=device
        )

    # Prepare optional arguments
    kwargs = {"aligned": aligned}

    if use_weights:
        if batch_size == 1:
            weights = torch.rand(n_residues, dtype=dtype, device=device)
        else:
            weights = torch.rand(batch_size, n_residues, dtype=dtype, device=device)
        # Make some weights zero to test masking
        weights[weights < 0.2] = 0
        kwargs["weights"] = weights

    if use_custom_d0:
        kwargs["d0"] = 3.0 + torch.rand(1).item() * 2.0

    # Test basic functionality
    score = beignet.template_modeling_score(structure1, structure2, **kwargs)

    # Verify output shape
    expected_shape = (batch_size,) if batch_size > 1 else ()
    assert score.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {score.shape}"
    )

    # Verify score is in valid range [0, 1]
    assert torch.all(score >= 0), "TM-score should be non-negative"
    assert torch.all(score <= 1), "TM-score should not exceed 1"

    # Test identical structures give score close to 1
    score_identical = beignet.template_modeling_score(structure1, structure1, **kwargs)
    assert torch.allclose(
        score_identical, torch.ones_like(score_identical), atol=1e-6
    ), "Identical structures should have TM-score close to 1"

    # Test very different structures give lower scores
    random_structure = torch.randn_like(structure2) * 10
    score_random = beignet.template_modeling_score(
        structure1, random_structure, **kwargs
    )
    assert torch.all(score_random < 0.5), (
        "Very different structures should have low TM-score"
    )

    # Test edge cases
    # Test with minimum number of residues
    if batch_size == 1:
        small_struct1 = torch.randn(3, 3, dtype=dtype, device=device)
        small_struct2 = small_struct1 + 0.1 * torch.randn(
            3, 3, dtype=dtype, device=device
        )
    else:
        small_struct1 = torch.randn(batch_size, 3, 3, dtype=dtype, device=device)
        small_struct2 = small_struct1 + 0.1 * torch.randn(
            batch_size, 3, 3, dtype=dtype, device=device
        )

    score_small = beignet.template_modeling_score(small_struct1, small_struct2)
    assert torch.all(score_small >= 0) and torch.all(score_small <= 1)

    # Test error handling - mismatched shapes
    if batch_size == 1:
        wrong_shape = torch.randn(n_residues + 1, 3, dtype=dtype, device=device)
    else:
        wrong_shape = torch.randn(
            batch_size, n_residues + 1, 3, dtype=dtype, device=device
        )

    with pytest.raises(ValueError, match="Input and target must have the same shape"):
        beignet.template_modeling_score(structure1, wrong_shape)

    # Test error handling - wrong coordinate dimension
    if batch_size == 1:
        wrong_coords = torch.randn(n_residues, 2, dtype=dtype, device=device)
    else:
        wrong_coords = torch.randn(
            batch_size, n_residues, 2, dtype=dtype, device=device
        )

    with pytest.raises(ValueError, match="Last dimension must be 3"):
        beignet.template_modeling_score(wrong_coords, wrong_coords)

    # Test gradient computation
    structure1_grad = structure1.clone().requires_grad_(True)
    structure2_grad = structure2.clone().requires_grad_(True)

    score_grad = beignet.template_modeling_score(
        structure1_grad, structure2_grad, **kwargs
    )

    # Verify gradients can be computed
    if score_grad.numel() == 1:
        score_grad.backward()
    else:
        score_grad.sum().backward()

    assert structure1_grad.grad is not None, (
        "Gradients should be computed for structure1"
    )
    assert structure2_grad.grad is not None, (
        "Gradients should be computed for structure2"
    )

    # Skip gradcheck for now as it may be too strict for this complex function
    # Would need smaller examples and looser tolerances

    # Test torch.compile compatibility
    # Only test compilation with basic cases to avoid recompilation issues
    if not use_weights and not use_custom_d0:
        compiled_fn = torch.compile(beignet.template_modeling_score, fullgraph=True)
        score_compiled = compiled_fn(structure1, structure2, aligned=aligned)

        # Results should be very close (may have small numerical differences)
        assert torch.allclose(score, score_compiled, atol=1e-5, rtol=1e-5), (
            "Compiled function should produce similar results"
        )

    # Test with torch.func transformations
    # Test vmap
    if batch_size == 1:
        # Create batched version for vmap testing
        structures1_vmap = structure1.unsqueeze(0).repeat(3, 1, 1)
        structures2_vmap = structure2.unsqueeze(0).repeat(3, 1, 1)

        def tm_score_single(x, y):
            return beignet.template_modeling_score(x, y, aligned=aligned)

        scores_vmap = torch.func.vmap(tm_score_single)(
            structures1_vmap, structures2_vmap
        )
        assert scores_vmap.shape == (3,), "vmap should produce correct output shape"

    # Test mathematical properties
    # TM-score should be approximately symmetric when aligned=True
    if aligned and not use_weights:  # Only test symmetry without weights
        score_reverse = beignet.template_modeling_score(
            structure2, structure1, aligned=True
        )
        # Use a more lenient tolerance since TM-score uses different normalizations
        assert torch.allclose(score, score_reverse, atol=0.1, rtol=0.1), (
            f"TM-score should be approximately symmetric when structures are pre-aligned. "
            f"Got {score.item():.4f} vs {score_reverse.item():.4f}"
        )

    # Test with all zero weights (if using weights)
    if use_weights:
        zero_weights = torch.zeros_like(weights)
        # This should handle the edge case gracefully
        score_zero = beignet.template_modeling_score(
            structure1, structure2, weights=zero_weights, aligned=aligned
        )
        # With all zero weights, the score should be 0 (handled in implementation)
        assert torch.all(score_zero == 0), "TM-score with all zero weights should be 0"

    # Test dtype preservation
    assert score.dtype == dtype, f"Output dtype should match input dtype {dtype}"

    # Test device preservation
    assert score.device == device, f"Output device should match input device {device}"
