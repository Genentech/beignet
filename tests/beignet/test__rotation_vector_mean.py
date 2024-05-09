import beignet
import hypothesis.strategies
import torch.testing
from scipy.spatial.transform import Rotation


@hypothesis.strategies.composite
def _strategy(function):
    size = function(
        hypothesis.strategies.integers(
            min_value=16,
            max_value=32,
        ),
    )

    input = Rotation.random(size)

    degrees = function(hypothesis.strategies.booleans())

    return (
        {
            "input": torch.from_numpy(input.as_rotvec(degrees)),
            "degrees": degrees,
        },
        torch.unsqueeze(
            torch.from_numpy(input.mean().as_rotvec(degrees)),
            dim=0,
        ),
    )


@hypothesis.given(_strategy())
def test_rotation_vector_mean(data):
    parameters, expected = data

    torch.testing.assert_close(
        beignet.rotation_vector_mean(**parameters),
        expected,
    )
