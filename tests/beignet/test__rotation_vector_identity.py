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

    degrees = function(hypothesis.strategies.booleans())

    return (
        {
            "size": size,
            "dtype": torch.float64,
        },
        torch.from_numpy(rotation.as_rotvec(degrees)),
    )


@hypothesis.given(_strategy())
def test_rotation_vector_identity(data):
    parameters, expected = data

    torch.testing.assert_close(
        beignet.rotation_vector_identity(**parameters),
        expected,
    )
