import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import beignet


@given(
    n_atoms=st.integers(min_value=10, max_value=100),
    batch_size=st.integers(min_value=1, max_value=4),
    cutoff=st.floats(min_value=5.0, max_value=20.0),
    dtype=st.sampled_from([torch.float32, torch.float64]),
    per_atom=st.booleans(),
    thresholds=st.just([0.5, 1.0, 2.0, 4.0]),  # Standard LDDT thresholds in Angstroms
)
@settings(max_examples=20, deadline=None)
def test_local_distance_difference_test(
    n_atoms, batch_size, cutoff, dtype, per_atom, thresholds
):
    """Test local_distance_difference_test operator with comprehensive scenarios."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate predicted and reference coordinates
    predicted_coords = (
        torch.randn(batch_size, n_atoms, 3, dtype=dtype, device=device) * 10
    )
    # Reference coords are slightly perturbed from predicted
    reference_coords = predicted_coords + torch.randn_like(predicted_coords) * 0.5

    # Create a mask for valid atoms (some atoms might be missing)
    atom_mask = torch.rand(batch_size, n_atoms, device=device) > 0.1

    # Calculate LDDT
    lddt_scores = beignet.local_distance_difference_test(
        predicted_coords=predicted_coords,
        reference_coords=reference_coords,
        atom_mask=atom_mask,
        cutoff=cutoff,
        thresholds=thresholds,
        per_atom=per_atom,
    )

    # Verify output shape
    if per_atom:
        assert lddt_scores.shape == (batch_size, n_atoms), (
            f"Expected shape {(batch_size, n_atoms)}, got {lddt_scores.shape}"
        )
    else:
        assert lddt_scores.shape == (batch_size,), (
            f"Expected shape {(batch_size,)}, got {lddt_scores.shape}"
        )

    # Verify scores are in [0, 1] range
    assert torch.all(lddt_scores >= 0) and torch.all(lddt_scores <= 1), (
        "LDDT scores should be in [0, 1] range"
    )

    # Test with perfect prediction (same coordinates)
    perfect_lddt = beignet.local_distance_difference_test(
        predicted_coords=reference_coords,
        reference_coords=reference_coords,
        atom_mask=atom_mask,
        cutoff=cutoff,
        thresholds=thresholds,
        per_atom=per_atom,
    )

    # Perfect prediction should have high scores (close to 1)
    assert torch.all(perfect_lddt > 0.99), (
        f"Perfect prediction should have LDDT close to 1, got {perfect_lddt}"
    )

    # Test with very bad prediction (large random coords)
    bad_predicted = torch.randn_like(predicted_coords) * 100
    bad_lddt = beignet.local_distance_difference_test(
        predicted_coords=bad_predicted,
        reference_coords=reference_coords,
        atom_mask=atom_mask,
        cutoff=cutoff,
        thresholds=thresholds,
        per_atom=per_atom,
    )

    # Bad prediction should have low scores
    mean_bad_score = bad_lddt.mean().item()
    assert mean_bad_score < 0.3, (
        f"Bad prediction should have low LDDT, got {mean_bad_score}"
    )

    # Note: LDDT uses hard thresholds which makes it non-differentiable
    # So we don't test gradients here

    # Test with different threshold values
    strict_thresholds = [0.25, 0.5, 1.0, 2.0]
    lenient_thresholds = [1.0, 2.0, 4.0, 8.0]

    strict_lddt = beignet.local_distance_difference_test(
        predicted_coords=predicted_coords,
        reference_coords=reference_coords,
        atom_mask=atom_mask,
        cutoff=cutoff,
        thresholds=strict_thresholds,
        per_atom=False,
    )

    lenient_lddt = beignet.local_distance_difference_test(
        predicted_coords=predicted_coords,
        reference_coords=reference_coords,
        atom_mask=atom_mask,
        cutoff=cutoff,
        thresholds=lenient_thresholds,
        per_atom=False,
    )

    # Lenient thresholds should give higher scores
    assert torch.all(lenient_lddt >= strict_lddt - 1e-6), (
        "Lenient thresholds should give equal or higher scores"
    )

    # Test torch.compile compatibility
    if n_atoms < 50:  # Only compile for smaller sizes to avoid timeout
        compiled_fn = torch.compile(
            beignet.local_distance_difference_test, fullgraph=True
        )
        lddt_compiled = compiled_fn(
            predicted_coords=predicted_coords,
            reference_coords=reference_coords,
            atom_mask=atom_mask,
            cutoff=cutoff,
            thresholds=thresholds,
            per_atom=per_atom,
        )

        assert torch.allclose(lddt_scores, lddt_compiled, atol=1e-5, rtol=1e-5), (
            "Compiled function should produce same results"
        )

    # Test dtype preservation
    assert lddt_scores.dtype == dtype, f"Output dtype should match input dtype {dtype}"

    # Test device preservation
    assert lddt_scores.device == device, (
        f"Output device should match input device {device}"
    )

    # Test error handling
    with pytest.raises(ValueError, match="must have the same shape"):
        beignet.local_distance_difference_test(
            predicted_coords=predicted_coords[..., :2],  # Wrong shape
            reference_coords=reference_coords,
            atom_mask=atom_mask,
            cutoff=cutoff,
            thresholds=thresholds,
        )

    with pytest.raises(ValueError, match="Thresholds must be positive"):
        beignet.local_distance_difference_test(
            predicted_coords=predicted_coords,
            reference_coords=reference_coords,
            atom_mask=atom_mask,
            cutoff=cutoff,
            thresholds=[-1.0, 1.0, 2.0],
        )

    # Test symmetry: LDDT should be symmetric w.r.t coordinate transformations
    # Apply a random rotation and translation
    rotation = torch.randn(3, 3, dtype=dtype, device=device)
    rotation = torch.linalg.qr(rotation)[0]  # Make it a proper rotation matrix
    translation = torch.randn(3, dtype=dtype, device=device)

    # Transform both predicted and reference
    pred_transformed = predicted_coords @ rotation.T + translation
    ref_transformed = reference_coords @ rotation.T + translation

    lddt_transformed = beignet.local_distance_difference_test(
        predicted_coords=pred_transformed,
        reference_coords=ref_transformed,
        atom_mask=atom_mask,
        cutoff=cutoff,
        thresholds=thresholds,
        per_atom=per_atom,
    )

    assert torch.allclose(lddt_scores, lddt_transformed, atol=1e-5, rtol=1e-5), (
        "LDDT should be invariant to rigid transformations"
    )
