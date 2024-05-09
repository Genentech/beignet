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
        },
        torch.from_numpy(rotations.inv().as_rotvec(degrees)),
    )


@hypothesis.given(_strategy())
def test_invert_rotation_vector(data):
    parameters, expected = data

    torch.testing.assert_close(
        beignet.invert_rotation_vector(**parameters),
        expected,
    )
