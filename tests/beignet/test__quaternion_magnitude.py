import beignet
import hypothesis.strategies
import torch.testing
from scipy.spatial.transform import Rotation


@hypothesis.strategies.composite
def _strategy(function):
    canonical = function(hypothesis.strategies.booleans())

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
            "input": torch.from_numpy(rotations.as_quat(canonical)),
            "canonical": canonical,
        },
        torch.from_numpy(rotations.magnitude()),
    )


@hypothesis.given(_strategy())
def test_quaternion_magnitude(data):
    parameters, expected = data

    torch.testing.assert_close(
        beignet.quaternion_magnitude(**parameters),
        expected,
    )
