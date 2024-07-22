import beignet
import hypothesis.strategies
import torch.testing
from scipy.spatial.transform import Rotation


@hypothesis.strategies.composite
def _strategy(function):
    degrees = function(hypothesis.strategies.booleans())

    rotations = Rotation.random(
        function(
            hypothesis.strategies.integers(
                min_value=1,
                max_value=8,
            ),
        ),
    )

    return (
        {
            "input": torch.from_numpy(rotations.as_rotvec(degrees)),
            "degrees": degrees,
        },
        torch.from_numpy(rotations.magnitude()),
    )


@hypothesis.given(_strategy())
def test_rotation_vector_magnitude(data):
    parameters, expected = data

    torch.testing.assert_close(
        beignet.rotation_vector_magnitude(**parameters),
        expected,
    )
