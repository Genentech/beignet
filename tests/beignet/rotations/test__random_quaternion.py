import hypothesis.strategies
import torch

import beignet.rotations


@hypothesis.strategies.composite
def _strategy(function):
    size = function(
        hypothesis.strategies.integers(
            min_value=1,
            max_value=8,
        ),
    )

    canonical = function(hypothesis.strategies.booleans())

    return (
        {
            "size": size,
            "canonical": canonical,
        },
        None,
    )


@hypothesis.given(_strategy())
def test_random_quaternion(data):
    """Test that random_quaternion generates normalized quaternions with correct properties."""
    parameters, _ = data

    quaternions = beignet.rotations.random_quaternion(**parameters)

    assert quaternions.shape == (parameters["size"], 4)

    norm = torch.norm(quaternions, dim=-1)
    torch.testing.assert_close(norm, torch.ones_like(norm), atol=1e-6, rtol=1e-6)
