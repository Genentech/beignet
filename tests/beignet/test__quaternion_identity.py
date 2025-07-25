import hypothesis.strategies
import torch
from scipy.spatial.transform import Rotation

import beignet


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
        torch.from_numpy(rotation.as_quat()),
    )


@hypothesis.given(_strategy())
def test_quaternion_identity(data):
    parameters, expected = data

    torch.testing.assert_close(
        beignet.quaternion_identity(**parameters),
        expected,
    )
