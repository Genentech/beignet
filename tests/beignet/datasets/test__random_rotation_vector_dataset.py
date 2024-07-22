import beignet
import hypothesis.strategies
from beignet.datasets import RandomRotationVectorDataset


@hypothesis.strategies.composite
def _strategy(function):
    size = function(
        hypothesis.strategies.integers(
            min_value=1,
            max_value=8,
        ),
    )

    degrees = function(hypothesis.strategies.booleans())

    return (
        {
            "size": size,
            "degrees": degrees,
        },
        beignet.random_rotation_vector(size, degrees=degrees),
    )


class TestRandomRotationVectorDataset:
    @hypothesis.given(_strategy())
    def test___init__(self, data):
        parameters, output = data

        dataset = RandomRotationVectorDataset(**parameters)

        assert dataset.data.shape == output.shape

        assert dataset.data.dtype == output.dtype

        assert dataset.data.layout == output.layout
