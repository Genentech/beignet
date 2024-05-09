import beignet
import hypothesis.strategies
import torch.testing
from scipy.spatial.transform import Rotation


@hypothesis.strategies.composite
def _strategy(function):
    rotation = Rotation.random(
        function(
            hypothesis.strategies.integers(
                min_value=1,
                max_value=8,
            ),
        ),
    )

    return (
        {
            "input": torch.from_numpy(
                rotation.as_quat(
                    canonical=False,
                ),
            ),
        },
        torch.unsqueeze(
            torch.abs(
                torch.from_numpy(
                    rotation.mean().as_quat(
                        canonical=False,
                    ),
                ),
            ),
            dim=0,
        ),
    )


@hypothesis.given(_strategy())
def test_quaternion_mean(data):
    parameters, expected = data

    torch.testing.assert_close(
        torch.abs(
            beignet.quaternion_mean(
                **parameters,
            ),
        ),
        expected,
    )
