import hypothesis.strategies
import torch
from scipy.spatial.transform import Rotation

import beignet


@hypothesis.strategies.composite
def _strategy(function):
    rotation = Rotation.random(
        function(
            hypothesis.strategies.integers(
                min_value=16,
                max_value=32,
            ),
        ),
    )

    return (
        {
            "input": torch.from_numpy(
                rotation.as_quat(
                    canonical=False,
                ),
            ),
        },
        torch.from_numpy(
            rotation.as_matrix(),
        ),
    )


@hypothesis.given(_strategy())
def test_quaternion_to_rotation_matrix(data):
    """Test quaternion to rotation matrix conversion with basic properties."""
    parameters, expected = data

    # Test with normalized quaternions from scipy
    torch.testing.assert_close(
        beignet.quaternion_to_rotation_matrix(**parameters),
        expected,
    )

    unnormalized_quat = parameters["input"] * 2.5  # Scale it
    result_unnormalized = beignet.quaternion_to_rotation_matrix(unnormalized_quat)

    # Should produce the same result as normalized quaternions
    torch.testing.assert_close(result_unnormalized, expected, atol=1e-5, rtol=1e-5)

    determinants = torch.det(result_unnormalized)
    torch.testing.assert_close(
        determinants, torch.ones_like(determinants), atol=1e-5, rtol=1e-5
    )
