import beignet
import hypothesis.strategies
from beignet.datasets import RandomRotationMatrixDataset


@hypothesis.strategies.composite
def _strategy(function):
    size = function(
        hypothesis.strategies.integers(
            min_value=1,
            max_value=8,
        ),
    )

    return (
        {
            "size": size,
        },
        beignet.random_rotation_matrix(size),
    )


class TestRandomRotationMatrixDataset:
    @hypothesis.given(_strategy())
    def test___init__(self, data):
        parameters, output = data

        dataset = RandomRotationMatrixDataset(**parameters)

        assert dataset.data.shape == output.shape

        assert dataset.data.dtype == output.dtype

        assert dataset.data.layout == output.layout
