import hypothesis.strategies
import torch
import torch.testing

import beignet


@hypothesis.strategies.composite
def _strategy(function):
    input = torch.randn(
        [
            function(
                hypothesis.strategies.integers(
                    min_value=1,
                    max_value=8,
                ),
            ),
            3,
        ]
    )

    transform = torch.randn([3, 3])

    return (
        (input, transform),
        beignet.apply_transform(
            beignet.apply_transform(input, transform),
            beignet.invert_transform(transform),
        ),
    )


@hypothesis.given(_strategy())
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture]
)
def test_invert_transform(float64, data):
    (input, transform), expected = data

    torch.testing.assert_close(
        input,
        beignet.apply_transform(
            beignet.apply_transform(input, transform),
            beignet.invert_transform(transform),
        ),
    )
