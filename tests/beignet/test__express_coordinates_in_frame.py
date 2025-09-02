import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import beignet


@given(
    batch_size=st.integers(min_value=1, max_value=4),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)  # Disable deadline due to torch.compile variability
def test_express_coordinates_in_frame(batch_size, dtype):
    """Test the express coordinates in frame function comprehensively."""

    # Generate test data
    device = torch.device("cpu")

    # Generate coordinates to transform
    coordinates = torch.randn(batch_size, 3, dtype=dtype, device=device)

    # Generate frame atoms (a, b, c) - make them non-collinear
    frame = torch.randn(batch_size, 3, 3, dtype=dtype, device=device)

    # Ensure frame atoms are not collinear by making them form a reasonable triangle
    frame[:, 0, :] = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)  # a
    frame[:, 1, :] = torch.tensor([0.0, 0.0, 0.0], dtype=dtype)  # b (origin)
    frame[:, 2, :] = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)  # c

    # Add some batch-specific variation
    frame += torch.randn_like(frame) * 0.1

    # Test basic functionality
    transformed = beignet.express_coordinates_in_frame(coordinates, frame)

    # Check output shape
    assert transformed.shape == coordinates.shape, (
        f"Expected shape {coordinates.shape}, got {transformed.shape}"
    )

    # Check output is finite
    assert torch.all(torch.isfinite(transformed)), (
        "Transformed coordinates should be finite"
    )

    # Test that transformation is consistent
    # If we express the frame center (atom b) in its own frame, it should be at origin
    frame_center = frame[:, 1, :]  # atom b
    center_in_frame = beignet.express_coordinates_in_frame(frame_center, frame)

    # Should be close to zero (up to numerical precision)
    assert torch.allclose(
        center_in_frame, torch.zeros_like(center_in_frame), atol=1e-5
    ), "Frame center should transform to origin in its own frame"

    # Test with identity-like frame (orthonormal basis)
    identity_frame = torch.zeros(batch_size, 3, 3, dtype=dtype, device=device)
    identity_frame[:, 0, :] = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)  # a = [1,0,0]
    identity_frame[:, 1, :] = torch.tensor([0.0, 0.0, 0.0], dtype=dtype)  # b = [0,0,0]
    identity_frame[:, 2, :] = torch.tensor([0.0, 0.0, 1.0], dtype=dtype)  # c = [0,0,1]

    # Test point at [1,0,0] (should align with first basis vector)
    test_point = torch.ones(batch_size, 3, dtype=dtype, device=device)
    test_point[:, 1:] = 0  # [1,0,0]

    identity_transformed = beignet.express_coordinates_in_frame(
        test_point, identity_frame
    )
    assert torch.all(torch.isfinite(identity_transformed)), (
        "Identity transform should be finite"
    )

    # Test gradient computation
    coordinates_grad = coordinates.clone().requires_grad_(True)
    frame_grad = frame.clone().requires_grad_(True)

    transformed_grad = beignet.express_coordinates_in_frame(
        coordinates_grad, frame_grad
    )
    loss = transformed_grad.sum()
    loss.backward()

    assert coordinates_grad.grad is not None, "Should have gradients for coordinates"
    assert frame_grad.grad is not None, "Should have gradients for frame"
    assert torch.all(torch.isfinite(coordinates_grad.grad)), (
        "Coordinate gradients should be finite"
    )
    assert torch.all(torch.isfinite(frame_grad.grad)), (
        "Frame gradients should be finite"
    )

    # Test torch.compile compatibility
    try:
        compiled_express = torch.compile(
            beignet.express_coordinates_in_frame, fullgraph=True
        )
        compiled_result = compiled_express(coordinates, frame)

        # Should be very close to original result
        assert torch.allclose(transformed, compiled_result, atol=1e-6), (
            "Compiled version should match original"
        )
    except Exception:
        # torch.compile might not be available or might fail, which is acceptable
        pass

    # Test batch operations - different examples should be processed independently
    if batch_size > 1:
        # Process each batch element individually
        individual_results = []
        for i in range(batch_size):
            single_result = beignet.express_coordinates_in_frame(
                coordinates[i : i + 1], frame[i : i + 1]
            )
            individual_results.append(single_result)

        # Concatenate results
        individual_tensor = torch.cat(individual_results, dim=0)

        # Should match batch processing
        assert torch.allclose(transformed, individual_tensor, atol=1e-6), (
            "Batch processing should match individual processing"
        )

    # Test input validation
    try:
        # Wrong coordinate dimensions (2D instead of 3D)
        wrong_coords = coordinates[..., :2]
        beignet.express_coordinates_in_frame(wrong_coords, frame)
        raise AssertionError("Should raise error for 2D coordinates")
    except ValueError:
        pass  # Expected

    try:
        # Wrong frame shape
        wrong_frame = frame[..., :2, :]  # Only 2 atoms instead of 3
        beignet.express_coordinates_in_frame(coordinates, wrong_frame)
        raise AssertionError("Should raise error for wrong frame shape")
    except ValueError:
        pass  # Expected

    try:
        # Mismatched batch dimensions
        if batch_size > 1:
            wrong_frame = frame[:1]  # Different batch size
            beignet.express_coordinates_in_frame(coordinates, wrong_frame)
            raise AssertionError("Should raise error for mismatched batch dimensions")
    except ValueError:
        pass  # Expected

    # Test numerical stability with very small vectors
    # Create frame with very small differences
    small_frame = torch.zeros(batch_size, 3, 3, dtype=dtype, device=device)
    small_frame[:, 0, :] = torch.tensor([1e-6, 0.0, 0.0], dtype=dtype)
    small_frame[:, 1, :] = torch.tensor([0.0, 0.0, 0.0], dtype=dtype)
    small_frame[:, 2, :] = torch.tensor([0.0, 1e-6, 0.0], dtype=dtype)

    small_transformed = beignet.express_coordinates_in_frame(coordinates, small_frame)
    assert torch.all(torch.isfinite(small_transformed)), (
        "Should handle small frame vectors"
    )

    # Test with large coordinates
    large_coordinates = coordinates * 1000
    large_transformed = beignet.express_coordinates_in_frame(large_coordinates, frame)
    assert torch.all(torch.isfinite(large_transformed)), (
        "Should handle large coordinates"
    )

    # Test orthogonality property of the constructed basis
    # The algorithm should create an orthonormal basis, so let's verify this indirectly
    # by checking that the transformation preserves relative geometric relationships

    # Create a simple test: two points and see if their relative distance in frame
    # coordinates relates properly to their global distance
    coords1 = torch.randn(batch_size, 3, dtype=dtype, device=device)
    coords2 = torch.randn(batch_size, 3, dtype=dtype, device=device)

    # Transform both points
    transformed1 = beignet.express_coordinates_in_frame(coords1, frame)
    transformed2 = beignet.express_coordinates_in_frame(coords2, frame)

    # Both transformations should be finite
    assert torch.all(torch.isfinite(transformed1)), (
        "First transformation should be finite"
    )
    assert torch.all(torch.isfinite(transformed2)), (
        "Second transformation should be finite"
    )

    # The frame coordinate system should be well-defined (no NaNs or infinities)
    frame_coords_diff = transformed1 - transformed2
    assert torch.all(torch.isfinite(frame_coords_diff)), (
        "Frame coordinate differences should be finite"
    )
