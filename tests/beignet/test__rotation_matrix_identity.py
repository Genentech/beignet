import beignet
import hypothesis.strategies
import torch
from scipy.spatial.transform import Rotation


@hypothesis.strategies.composite
def _strategy(function):
    size = function(
        hypothesis.strategies.integers(
            min_value=1,
            max_value=8,
        ),
    )

    rotation = Rotation.identity(size)

    return (
        {
            "size": size,
            "dtype": torch.float64,
        },
        torch.from_numpy(rotation.as_matrix()),
    )


@hypothesis.given(_strategy())
def test_rotation_matrix_identity(data):
    parameters, expected = data

    torch.testing.assert_close(
        beignet.rotation_matrix_identity(**parameters),
        expected,
    )
