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

    canonical = function(hypothesis.strategies.booleans())

    return (
        {
            "size": size,
            "canonical": canonical,
        },
        None,
    )


@hypothesis.given(_strategy())
def test_random_quaternion(data):
    parameters, _ = data

    assert beignet.random_quaternion(
        **parameters,
    ).shape == (parameters["size"], 4)
