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

    return (
        {
            "input": torch.from_numpy(input.as_matrix()),
        },
        torch.unsqueeze(torch.from_numpy(input.mean().as_matrix()), dim=0),
    )


@hypothesis.given(_strategy())
def test_rotation_matrix_mean(data):
    parameters, expected = data

    torch.testing.assert_close(
        beignet.rotation_matrix_mean(**parameters),
        expected,
    )
