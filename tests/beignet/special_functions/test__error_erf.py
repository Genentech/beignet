import hypothesis
import hypothesis.strategies
import scipy
import torch

import beignet.special_functions


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
            steps=128,
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
            steps=128,
            dtype=torch.float64,
        ),
        indexing="xy",
    )

    input = x + 1.0j * y

    # Call SciPy with NumPy input to avoid NumPy __array_wrap__ deprecation warnings
    return input, torch.as_tensor(scipy.special.erf(input.numpy()))


@hypothesis.given(_strategy())
def test_error_erf(data):
    input, output = data

    torch.testing.assert_close(beignet.special_functions.error_erf(input), output)
