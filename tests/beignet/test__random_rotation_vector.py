import beignet
import hypothesis.strategies


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
        None,
    )


@hypothesis.given(_strategy())
def test_random_rotation_vector(data):
    parameters, _ = data

    assert beignet.random_rotation_vector(
        **parameters,
    ).shape == (parameters["size"], 3)
