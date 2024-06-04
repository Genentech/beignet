import beignet
import hypothesis.strategies
from beignet.datasets import RandomQuaternionDataset


@hypothesis.strategies.composite
def _strategy(function):
    size = function(
        hypothesis.strategies.integers(
            min_value=1,
            max_value=8,
        ),
    )

    canonical = function(hypothesis.strategies.booleans())

    return (
        {
            "size": size,
            "canonical": canonical,
        },
        beignet.random_quaternion(size, canonical=canonical),
    )


class TestRandomQuaternionDataset:
    @hypothesis.given(_strategy())
    def test___init__(self, data):
        parameters, output = data

        dataset = RandomQuaternionDataset(**parameters)

        assert dataset.data.shape == output.shape

        assert dataset.data.dtype == output.dtype

        assert dataset.data.layout == output.layout
