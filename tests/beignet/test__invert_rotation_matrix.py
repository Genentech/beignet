import beignet
import hypothesis.strategies
import torch.testing
from scipy.spatial.transform import Rotation


@hypothesis.strategies.composite
def _strategy(function):
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
            "input": torch.from_numpy(rotations.as_matrix()),
        },
        torch.from_numpy(rotations.inv().as_matrix()),
    )


@hypothesis.given(_strategy())
def test_invert_rotation_matrix(data):
    parameters, expected = data

    torch.testing.assert_close(
        beignet.invert_rotation_matrix(**parameters),
        expected,
    )
