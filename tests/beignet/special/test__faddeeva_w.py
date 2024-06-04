import beignet.special
import hypothesis
import hypothesis.strategies
import scipy
import torch


@hypothesis.strategies.composite
def _strategy(function):
    x, y = torch.meshgrid(
        torch.linspace(
            function(
                hypothesis.strategies.floats(
                    min_value=-10,
                    max_value=-10,
                ),
            ),
            function(
                hypothesis.strategies.floats(
                    min_value=10,
                    max_value=10,
                ),
            ),
            steps=function(
                hypothesis.strategies.integers(
                    min_value=128,
                    max_value=512,
                ),
            ),
            dtype=torch.float64,
        ),
        torch.linspace(
            function(
                hypothesis.strategies.floats(
                    min_value=-10,
                    max_value=-10,
                ),
            ),
            function(
                hypothesis.strategies.floats(
                    min_value=10,
                    max_value=10,
                ),
            ),
            function(
                hypothesis.strategies.integers(
                    min_value=128,
                    max_value=512,
                ),
            ),
            dtype=torch.float64,
        ),
        indexing="xy",
    )

    input = x + 1.0j * y

    return input, scipy.special.wofz(input)


@hypothesis.given(_strategy())
def test_faddeeva_w(data):
    input, output = data

    torch.testing.assert_close(beignet.special.faddeeva_w(input), output)