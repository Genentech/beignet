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
        torch.from_numpy(rotations.magnitude()),
    )


@hypothesis.given(_strategy())
def test_rotation_matrix_magnitude(data):
    parameters, expected = data

    torch.testing.assert_close(
        beignet.rotation_matrix_magnitude(**parameters),
        expected,
    )
