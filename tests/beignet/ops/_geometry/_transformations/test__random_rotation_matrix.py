import beignet.ops
import hypothesis.strategies


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
        None,
    )


@hypothesis.given(_strategy())
def test_random_rotation_matrix(data):
    parameters, _ = data

    assert beignet.ops.random_rotation_matrix(
        **parameters,
    ).shape == (parameters["size"], 3, 3)
